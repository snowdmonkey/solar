from abc import ABC, abstractmethod
from typing import Tuple, Dict, List
from scipy.cluster.hierarchy import linkage, cut_tree
import numpy as np
import cv2
import sys


class PanelGroupLabeler(ABC):

    @abstractmethod
    def process_image(self, img_path: str) -> Dict[str, List[Tuple[int, int]]]:
        """
        process a given image
        :param img_path:
        :return: a dict group_id -> [(conner1_row, corner1_col), (conner2_row, corner2_col), (conner3_row, corner3_col),
         (conner4_row, corner4_col),...]
        """
        pass


class ColorBasedLabeler(PanelGroupLabeler):

    def process_image(self, img_path: str) -> Dict[str, List[Tuple[int, int]]]:
        """
        process a given image
        :param img_path:
        :return: a dict group_id -> [(conner1_row, corner1_col), (conner2_row, corner2_col), (conner3_row, corner3_col),
         (conner4_row, corner4_col)]
        """
        results = dict()
        raw_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        blue_scale = self._convert_blue_scale(raw_image)
        _, th = cv2.threshold(blue_scale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [x for x in contours if cv2.contourArea(x) > 10000]
        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            results.update({str(i): [(y, x), (y+h, x), (y+h, x+w), (y, x+w)]})
        # points = list()
        # for cnt in contours:
        #     for cnt_point in cnt:
        #         points.append(cnt_point[0])
        # points = np.array(points)
        # linkage_matrix = linkage(points, method='single', metric="chebyshev")
        # ctree = cut_tree(linkage_matrix, height=[10])
        # cluster = np.array([x[0] for x in ctree])
        # for i in range(max(cluster)+1):
        #     group_points = points[cluster == i]
        #     x, y, w, h = cv2.boundingRect(group_points)
        #     results.update({str(i): [(y, x), (y+h, x), (y, x+w), (y+h, x+w)]})
        return results

    @staticmethod
    def _convert_blue_scale(img: np.ndarray) -> np.ndarray:
        """
        convert a three-chanel image to one channel which indicates how blue a point is
        :param img: three-channel image
        :return: one_channel image
        """
        mask = img[:, :, 0] / (img[:, :, 1] / 3 + img[:, :, 2] / 3 + img[:, :, 0] / 3 + 0.001)
        mask = mask.astype(np.uint8)
        return mask


def main():
    img_path = sys.argv[1]
    labeler = ColorBasedLabeler()
    result = labeler.process_image(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    for k, v in result.items():
        cv2.rectangle(img, (v[0][1], v[0][0]), (v[2][1], v[2][0]), (0, 255, 0), 2)

    cv2.imwrite("labeled.png", img)


if __name__ == "__main__":
    main()