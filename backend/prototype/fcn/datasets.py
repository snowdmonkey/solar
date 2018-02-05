from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import logging
import cv2
import random
import torch


class FCNDataset(Dataset):

    # def __init__(self, feature_folder: str, label_folder: str):
    def __init__(self, *args: Tuple[Path, Path], color_map: Dict[Tuple[int, int, int], int], gray_scale: bool,
                 transform=None):
        """
        constructor
        :param args: tuples of (feature folder, annotation folder)
        :param color_map: a dict to transfer color(RGB) to int label
        :param gray_scale: whether to read feature map as gray scale, yes if true
        """
        self._path_pairs = list() # type: List[Tuple[Path, Path]]
        for arg in args:
            feature_folder, label_folder = arg
            for feature_path in feature_folder.glob("*.jpg"):
                label_path = label_folder / feature_path.name.replace(".jpg", ".png")
                if label_path.exists():
                    self._path_pairs.append((feature_path, label_path))

        self._logger = logging.getLogger(__name__)
        self._color_map = color_map
        self._gray_scale = gray_scale
        self._transform = transform
        self._logger.info("{} feature images included".format(len(self._path_pairs)))

    @property
    def n_channels(self) -> int:
        if self._gray_scale is True:
            return 1
        else:
            return 3

    def __len__(self):
        return len(self._path_pairs)

    def __getitem__(self, idx) -> Dict[str, np.ndarray]:

        feature_path, label_path = self._path_pairs[idx]

        if self._gray_scale:
            feature_img = cv2.imread(str(feature_path), cv2.IMREAD_GRAYSCALE)  # type: np.ndarray
            # feature_img = feature_img.reshape(1, feature_img.shape[0], feature_img.shape[1])
        else:
            feature_img = cv2.imread(str(feature_path), cv2.IMREAD_COLOR)  # type: np.ndarray
            # feature_img = np.rollaxis(feature_img, 2)

        # feature = (feature_img - 127.5) / 127.5

        label_img = cv2.imread(str(label_path), cv2.IMREAD_COLOR)  # type: np.ndarray
        label = np.zeros(shape=label_img.shape[:2], dtype=np.long)

        for k, v in self._color_map.items():
            label[(label_img[:, :, 0] == k[2]) & (label_img[:, :, 1] == k[1]) & (label_img[:, :, 0] == k[0])] = v

        sample = {"feature": feature_img, "label": label}

        if self._transform is not None:
            sample = self._transform(sample)

        return sample


class RandomScale:
    """
    random scale the image for training
    """
    def __init__(self, min_scale: float = 0.7, max_scale: float = 1.5):
        """
        constructor
        :param min_scale: minimum of the scale factor
        :param max_scale: maximum of the scale factor
        """
        self._min_scale = min_scale
        self._max_scale = max_scale

    def __call__(self, sample: Dict[str, np.ndarray]):
        feature, label = sample["feature"], sample["label"]

        scale_factor = random.uniform(self._min_scale, self._max_scale)

        feature = cv2.resize(feature, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        return {"feature": feature, "label": label}


class RandomCrop:
    """
    random crop the image for training
    """
    def __init__(self, height: int, width: int):
        """
        constructor
        :param height: height of the cropped image
        :param width: width of the cropped image
        """
        self._height = height
        self._width = width

    def __call__(self, sample: Dict[str, np.ndarray]):
        feature, label = sample["feature"], sample["label"]
        raw_height, raw_width = feature.shape[0], feature.shape[1]
        if self._height > raw_height:
            pad_top = (self._height - raw_height) // 2
            pad_btm = self._height - raw_height - pad_top
            feature = cv2.copyMakeBorder(feature, pad_top, pad_btm, 0, 0, borderType=cv2.BORDER_DEFAULT)
            label = cv2.copyMakeBorder(label, pad_top, pad_btm, 0, 0, borderType=cv2.BORDER_DEFAULT)

        if self._width > raw_width:
            pad_left = (self._width - raw_width) // 2
            pad_right = self._width - raw_width - pad_left
            feature = cv2.copyMakeBorder(feature, 0, 0, pad_left, pad_right, borderType=cv2.BORDER_DEFAULT)
            label = cv2.copyMakeBorder(label, 0, 0, pad_left, pad_right, borderType=cv2.BORDER_DEFAULT)

        top_index = random.choice(range(feature.shape[0]-self._height))
        left_index = random.choice(range(feature.shape[0]-self._width))

        feature = feature[top_index:(top_index+self._height), left_index:(left_index+self._width)]
        label = label[top_index:(top_index + self._height), left_index:(left_index + self._width)]

        return {"feature": feature, "label": label}


class ToTensor:
    """
    convert ndarrays to Tensors
    """

    def __call__(self, sample: Dict[str, np.ndarray]):
        feature, label = sample["feature"], sample["label"]

        if feature.ndim == 2:
            feature = feature.reshape(1, feature.shape[0], feature.shape[1])
        elif feature.ndim == 3:
            feature = np.rollaxis(feature, 2)
        else:
            raise Exception("unknown feature dimension {}".format(feature.shape))

        feature = (feature - 127.5) / 127.5

        return {"feature": torch.from_numpy(feature).float(),
                "label": torch.from_numpy(label).long()}
