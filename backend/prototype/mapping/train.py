import torch.utils.data
import torch.nn
import torch.optim
import os
import logging
import cv2
import numpy as np
import random
import argparse
from typing import Tuple, List

from torch.autograd import Variable

from fcn.nets import fc_dense_net57


class FCNDataset(torch.utils.data.Dataset):

    def __init__(self, feature_folder: str, label_folder: str):
        """
        constructor
        :param feature_folder: a folder store the feature image
        :param label_folder: a folder store the label image
        """
        feature_files = [f for f in os.listdir(feature_folder) if os.path.isfile(os.path.join(feature_folder, f))]
        label_files = [f for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))]
        self._files = [x for x in feature_files if x in label_files and x.lower().endswith((".png", "jpg"))]
        self._feature_folder = feature_folder
        self._label_folder = label_folder
        self._logger = logging.getLogger(__name__)

        self._logger.info("totally {} files found in feature folder".format(len(feature_files)))
        self._logger.info("totally {} files found in label folder".format(len(label_files)))
        self._logger.info("{} files will be used".format(len(self._files)))

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, str]:

        file_name = self._files[idx]

        feature_img = cv2.imread(os.path.join(self._feature_folder, file_name), cv2.IMREAD_COLOR)
        label_img = cv2.imread(os.path.join(self._label_folder, file_name), cv2.IMREAD_GRAYSCALE)

        feature = np.rollaxis(feature_img, 2)

        # label = np.zeros((2,)+label_img.shape)
        # label[0, label_img == 255] = 1
        # label[1, label_img == 0] = 1
        label = label_img/255

        return feature, label, file_name


class BigImageDataset(torch.utils.data.Dataset):
    """
    this data would use a very large image as feature and another very large image as the label. Each time it randomly
    crop a piece
    """

    def __init__(self, feature_image: str, label_image: str, crop_size: int, dataset_size: int):
        """
        constructor
        :param feature_image: image path of the big image
        :param label_image: image path of the label image
        :param crop_size: the size of the cropped sample
        :param dataset_size: totally how many samples to return
        """
        self._feature_image = cv2.imread(feature_image, cv2.IMREAD_COLOR)
        self._label_image = cv2.imread(label_image, cv2.IMREAD_GRAYSCALE)

        self._height, self._width = self._label_image.shape

        self._crop_size = crop_size
        self._dataset_size = dataset_size

    def __len__(self):
        return self._dataset_size

    def _get_random_block(self) -> Tuple[np.ndarray, np.ndarray]:
        left = random.randint(0, self._width-self._crop_size)
        top = random.randint(0, self._height-self._crop_size)
        right = left + self._crop_size
        btm = top + self._crop_size

        feature = self._feature_image[top:btm, left:right]
        label = self._label_image[top:btm, left:right]

        return feature, label

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:

        while True:
            feature, label = self._get_random_block()
            n_zero = (feature[:, :, 0] == 0).sum()
            zero_ratio = n_zero/label.size

            if zero_ratio < 0.5:
                break

        feature = np.rollaxis(feature, 2)/255
        label = label / 255

        feature = torch.from_numpy(feature).float()
        label = torch.from_numpy(label).long()
        return feature, label


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


def train(feature_image: str, label_image: str, n_epoch: int):

    train_dataset = BigImageDataset(feature_image, label_image, 400, 500)
    test_dataset = BigImageDataset(feature_image, label_image, 500, 100)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    net = fc_dense_net57(n_classes=2, channels=3)

    if torch.cuda.device_count() == 1:
        net.cuda()

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        net.cuda()

    criterion = torch.nn.NLLLoss2d()
    # optimizer = torch.optim.RMSprop(net.parameters())
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(n_epoch):
        running_loss = 0.0

        for i, data in enumerate(train_loader):

            feature, label = data

            if torch.cuda.is_available():
                feature = Variable(feature.cuda())
                label = Variable(label.cuda())
            else:
                feature = Variable(feature)
                label = Variable(label)

            optimizer.zero_grad()

            output = net(feature)

            loss = criterion(output, label)
            # loss = criterion.forward(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            # if i % 100 == 0:
            #
            #     logging.info("epoch: {}; batch: {}; running loss: {}".format(epoch, i, running_loss/100))
            #     running_loss = 0.0

        logging.info("epoch: {}; running loss: {}".format(epoch, running_loss / len(train_dataset)))

        net.eval()

        eval_model(net, test_loader)

        net.train()

        torch.save(net, "model.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_image", type=str, help="path of the feature image")
    parser.add_argument("label_image", type=str, help="path of the label image")
    parser.add_argument("--epoch", type=int, help="number of epochs to run", default=50)

    args = parser.parse_args()
    # print(args)
    train(args.feature_image, args.label_image, n_epoch=args.epoch)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler("log")])
    main()



