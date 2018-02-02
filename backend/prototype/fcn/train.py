import torch.utils.data
import torch.nn
import torch.optim
import os
import logging
import cv2
import numpy as np
import argparse
import random
from typing import Tuple, List
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
from fcn.nets import fc_dense_net57
from .datasets import FCNDataset
from tensorboardX import SummaryWriter


def list_mean(l: List[float]):

    return sum(l)/len(l)


def get_prediction(output: torch.FloatTensor) -> torch.LongTensor:
    bs, c, h, w = output.size()
    tensor = output.data
    _, indices = tensor.cpu().max(1)
    indices = indices.view(bs, h, w)
    return indices


def get_iou(pred: torch.LongTensor, label: torch.LongTensor, index: int) -> float:
    """
    compute the IoU over an index
    :param pred: predicted tensor
    :param label: label tensor
    :param index: index of interest
    :return: IoU
    """

    pred = pred.cpu()
    label = label.cpu()
    i_area = ((pred == index) & (label == index)).sum()
    u_area = ((pred == index) | (label == index)).sum()

    if u_area > 0:
        return i_area / u_area
    else:
        return 1.0


def get_acc(pred: torch.LongTensor, label: torch.LongTensor) -> float:
    """
    compute the accuracy of the prediction
    :param pred: prediction tensor
    :param label: label tensor
    :return: accuracy
    """
    assert pred.size() == label.size()
    pred = pred.cpu()
    label = label.cpu()
    n_acc = pred.eq(label).sum()
    acc_rate = n_acc / pred.numel()
    return acc_rate


class FCNTrainer:
    """
    class to train a DenseFCN model
    """

    def __init__(self, n_classes: int, n_channels: int):
        """
        constructor
        :param n_classes: number of classes
        :param n_channels: number of channels
        """
        self._n_classes = n_classes
        self._n_channels = n_channels
        self._net = fc_dense_net57(n_classes=n_classes, channels=n_channels)

        self._criterion = torch.nn.NLLLoss2d()
        # self._optimizer = torch.optim.SGD(self._net.parameters(), lr=0.01)
        self._optimizer = torch.optim.RMSprop(self._net.parameters(), lr=0.001)
        self._scheduler = StepLR(self._optimizer, step_size=1, gamma=0.995)

        if torch.cuda.device_count() == 1:
            self._net.cuda()

        if torch.cuda.device_count() > 1:
            self._net = torch.nn.DataParallel(self._net)
            self._net.cuda()

        self._logger = logging.getLogger(self.__class__.__name__)
        self._writer = SummaryWriter()

        self._train_loader = None
        self._test_loader = None

    def set_dataset(self, dataset: FCNDataset, train_proportion: float):

        if dataset.n_channels != self._n_channels:
            raise ValueError("n channels mismatch")

        if not 0.0 < train_proportion < 1.0:
            raise ValueError("train proportion should be between 0 and 1")

        indices = random.sample(range(len(dataset)), k=len(dataset))
        train_indices = indices[:int(len(dataset)*train_proportion)]
        test_indices = indices[int(len(dataset) * train_proportion):]

        train_sampler = SubsetRandomSampler(indices=train_indices)
        test_sampler = SubsetRandomSampler(indices=test_indices)

        if torch.cuda.device_count() < 2:
            batch_size = 1
        else:
            batch_size = torch.cuda.device_count()

        self._train_loader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=train_sampler)
        self._test_loader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=test_sampler)

    def _train(self, eval: bool) -> Tuple[float, float, float]:
        """
        train the model for one epoch
        :param: eval: true if eval model, false if train model
        :return: (loss, accuracy, iou)
        """
        if eval is True:
            self._net.eval()
            data_loader = self._test_loader
        else:
            self._net.train()
            data_loader = self._train_loader

        train_loss = 0.0
        train_acc = 0.0
        train_iou = 0.0

        for i, data in enumerate(data_loader):

            feature, label = data["feature"], data["label"]

            if torch.cuda.is_available():
                feature = Variable(feature.cuda())
                label = Variable(label.cuda())
            else:
                feature = Variable(feature)
                label = Variable(label)

            if eval is True:
                output = self._net(feature)
                loss = self._criterion(output, label)
            else:
                self._optimizer.zero_grad()
                output = self._net(feature)
                loss = self._criterion(output, label)
                loss.backward()
                self._optimizer.step()

            train_loss += loss.data[0]

            pred = get_prediction(output)
            train_acc += get_acc(pred, label.data)
            train_iou += get_iou(pred, label.data, 1)

            if eval is True:

                pred_folder = "./data/pre"
                if not os.path.exists(pred_folder):
                    os.makedirs(pred_folder)

                raw = feature.data.cpu().numpy() * 127.5 + 127.5
                raw = raw.round().astype(np.uint8)[0]
                raw = raw[0]
                raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

                # label_np = label.data.cpu().numpy()*100
                # label_np = label_np.astype(np.uint8)[0]
                # cv2.imwrite(os.path.join(pred_folder, "label{}.png".format(i)), label_np)

                pred_np = pred.cpu().numpy()[0]
                overlay = raw.copy()
                overlay[pred_np == 1] = (0, 255, 0)
                raw = cv2.addWeighted(raw, 0.7, overlay, 0.3, 0)
                #
                cv2.imwrite(os.path.join(pred_folder, "pred{}.png".format(i)), raw)

        if eval is False:
            self._scheduler.step()

        train_size = len(data_loader)
        return train_loss/train_size, train_acc/train_size, train_iou/train_size

    def _save(self):
        torch.save(self._net, "model.pt")

    def train(self, n_epochs: int):
        """
        train the model
        :param n_epochs: number of epochs to run, run infinitely if n_epochs=0
        :return: None
        """
        epoch = 0
        while True:
            epoch += 1

            if epoch > n_epochs != 0:
                break

            loss, acc, iou = self._train(eval=False)
            self._logger.info("epoch: {}; train loss: {}; train acc: {}; train iou: {}".format(epoch, loss, acc, iou))
            self._writer.add_scalar("data/loss/train", loss, epoch)
            self._writer.add_scalar("data/acc/train", acc, epoch)
            self._writer.add_scalar("data/iou/train", iou, epoch)

            loss, acc, iou = self._train(eval=True)
            self._logger.info("epoch: {};  test loss: {};  test acc: {};  test iou: {}".format(epoch, loss, acc, iou))
            self._writer.add_scalar("data/loss/test", loss, epoch)
            self._writer.add_scalar("data/acc/test", acc, epoch)
            self._writer.add_scalar("data/iou/test", iou, epoch)

            self._save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_image", type=str, help="path of the feature image")
    parser.add_argument("label_image", type=str, help="path of the label image")
    parser.add_argument("--epoch", type=int, help="number of epochs to run", default=50)

    args = parser.parse_args()
    # print(args)
    # train(args.feature_image, args.label_image, n_epoch=args.epoch)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler("log")])
    main()
