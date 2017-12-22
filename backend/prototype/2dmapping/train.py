import torch.utils.data
import torch.nn
import torch.optim
import os
import logging
import cv2
import numpy as np
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


def list_mean(l: List[float]):

    return sum(l)/len(l)


def eval_model(net: torch.nn.Module, data_loader: torch.utils.data.DataLoader):

    acc_list = list()

    pred_folder = "./data/pre"

    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)

    for i, data in enumerate(data_loader):

        feature, label, file_name = data
        feature = feature.float()
        label = label.long()

        if torch.cuda.is_available():
            feature = Variable(feature.cuda())
            label = Variable(label.cuda())
        else:
            feature = Variable(feature)
            label = Variable(label)

        output = net(feature)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).sum()
        acc = correct/torch.numel(feature.data)
        acc_list.append(acc)

        bs, c, h, w = output.size()
        _, indices = output.data.max(1)
        indices = indices.view(bs, h, w)
        output = indices.cpu().numpy()[0]
        th = np.uint8(output * 255)

        cv2.imwrite(os.path.join(pred_folder, file_name[0]), th)

    logging.info("test accuracy is {}".format(list_mean(acc_list)))


def main():
    train_dataset = FCNDataset("./data/feature", "./data/label")
    test_dataset = FCNDataset("./data/feature_test", "./data/label_test")

    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    net = fc_dense_net57(n_classes=2, channels=3)

    if torch.cuda.device_count() == 1:
        net.cuda()

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        net.cuda()

    criterion = torch.nn.NLLLoss2d()
    optimizer = torch.optim.RMSprop(net.parameters())

    for epoch in range(20):
        running_loss = 0.0

        for i, data in enumerate(train_loader):

            feature, label, _ = data
            feature = feature.float()
            label = label.long()

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

            if i % 100 == 0:

                logging.info("epoch: {}; batch: {}; running loss: {}".format(epoch, i, running_loss/100))
                running_loss = 0.0

        net.eval()

        eval_model(net, test_loader)

        net.train()

        torch.save(net, "model.pt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])
    main()



