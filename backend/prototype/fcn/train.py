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
from fcn.nets import fc_dense_net57
from .datasets import FCNDataset
from tensorboardX import SummaryWriter


def list_mean(l: List[float]):

    return sum(l)/len(l)


def get_iou(output: torch.LongTensor, label: torch.LongTensor, index: int) -> float:
    """
    compute the IoU over an index
    :param output: predicted tensor
    :param label: label tensor
    :param index: index of interest
    :return: IoU
    """
    i_area = ((output == index) & (label == index)).sum()
    u_area = ((output == index) | (label == index)).sum()

    if u_area > 0:
        return i_area/u_area
    else:
        return 1.0


def get_acc(output: torch.LongTensor, label: torch.LongTensor) -> float:
    """
    compute the accuracy of the prediction
    :param output: prediction tensor
    :param label: label tensor
    :return: accuracy
    """
    n_acc = output.eq(label).sum()
    acc_rate = n_acc / output.numel()
    return acc_rate


def eval_model(net: torch.nn.Module, data_loader: torch.utils.data.DataLoader):

    acc_list = list()
    iou_list = list()

    pred_folder = "./data/pre"

    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)

    for i, data in enumerate(data_loader):

        feature, label = data
        # feature = feature.float()
        # label = label.long()

        # raw = (feature.cpu().numpy()*255).astype(np.uint8)[0]
        raw = feature.cpu().numpy()*255
        raw = raw.round().astype(np.uint8)[0]
        raw = np.rollaxis(raw, 0, 3)
        # print(raw.shape)

        cv2.imwrite(os.path.join(pred_folder, "raw{}.png".format(i)), raw)

        if torch.cuda.is_available():
            feature = Variable(feature.cuda())
            label = Variable(label.cuda())
        else:
            feature = Variable(feature)
            label = Variable(label)

        output = net(feature)
        pred = output.data.max(1)[1]

        iou = get_iou(pred, label.data, 1)
        acc = get_acc(pred, label.data)

        # correct = pred.eq(label.data).sum()
        # acc = correct/torch.numel(feature.data)
        acc_list.append(acc)
        iou_list.append(iou)

        bs, c, h, w = output.size()
        _, indices = output.data.max(1)
        indices = indices.view(bs, h, w)
        output = indices.cpu().numpy()[0]
        th = np.uint8(output * 255)

        cv2.imwrite(os.path.join(pred_folder, "pred{}.png".format(i)), th)

    logging.info("test accuracy is {}, test iou is {}".format(list_mean(acc_list), list_mean(iou_list)))


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
        self._net = fc_dense_net57(n_classes=2, channels=3)
        if torch.cuda.device_count() == 1:
            self._net.cuda()

        if torch.cuda.device_count() > 1:
            self._net = torch.nn.DataParallel(self._net)
            self._net.cuda()

        self._logger = logging.getLogger(self.__class__.__name__)
        self._writer = SummaryWriter()

    # def eval(self, data_loader: DataLoader):
    #     acc_list = list()
    #     iou_list = list()
    #
    #     pred_folder = "./data/pre"
    #
    #     if not os.path.exists(pred_folder):
    #         os.makedirs(pred_folder)
    #
    #     for i, data in enumerate(data_loader):
    #
    #         feature, label = data["feature"], data["label"]
    #         # feature = feature.float()
    #         # label = label.long()
    #
    #         # raw = (feature.cpu().numpy()*255).astype(np.uint8)[0]
    #         # raw = feature.cpu().numpy() * 127.5 + 127.5
    #         # raw = raw.round().astype(np.uint8)[0]
    #         # raw = np.rollaxis(raw, 0, 3)
    #         # print(raw.shape)
    #
    #         # cv2.imwrite(os.path.join(pred_folder, "raw{}.png".format(i)), raw)
    #
    #         if torch.cuda.is_available():
    #             feature = Variable(feature.cuda())
    #             label = Variable(label.cuda())
    #         else:
    #             feature = Variable(feature)
    #             label = Variable(label)
    #
    #         output = self._net(feature)
    #         pred = output.data.max(1)[1]
    #
    #         iou = get_iou(pred, label.data, 1)
    #         acc = get_acc(pred, label.data)
    #
    #         # correct = pred.eq(label.data).sum()
    #         # acc = correct/torch.numel(feature.data)
    #         acc_list.append(acc)
    #         iou_list.append(iou)
    #
    #         # bs, c, h, w = output.size()
    #         # _, indices = output.data.max(1)
    #         # indices = indices.view(bs, h, w)
    #         # output = indices.cpu().numpy()[0]
    #         # th = np.uint8(output * 255)
    #
    #         # cv2.imwrite(os.path.join(pred_folder, "pred{}.png".format(i)), th)
    #
    #     self._logger.info("test accuracy is {}, test iou is {}".format(list_mean(acc_list), list_mean(iou_list)))
        # self._writer.add_scalar()

    def train(self, dataset: FCNDataset, n_epochs: int, train_proportion: float):
        """
        train the model
        :param dataset: fcn dataset
        :param n_epochs: number of epochs, if 0, train forever
        :param train_proportion: proportion for splitting training dataset
        :return: None
        """

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

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=test_sampler)

        epoch = 0

        criterion = torch.nn.NLLLoss2d()
        optimizer = torch.optim.SGD(self._net.parameters(), lr=0.01)

        while True:
            if epoch > n_epochs != 0:
                break
            else:
                epoch += 1

            running_loss = 0
            for data in train_loader:

                feature, label = data["feature"], data["label"]

                if torch.cuda.is_available():
                    feature = Variable(feature.cuda())
                    label = Variable(label.cuda())
                else:
                    feature = Variable(feature)
                    label = Variable(label)

                optimizer.zero_grad()
                output = self._net(feature)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]

            self._logger.info("epoch: {}; training loss: {}".format(epoch, running_loss/len(train_loader)))
            self._writer.add_scalar("data/train/loss", running_loss/len(train_loader), epoch)


            # self._net.eval()

            eval_model(self._net, test_loader)

            self._net.train()

            torch.save(self._net, "model.pt")


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