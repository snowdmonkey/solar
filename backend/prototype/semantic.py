from abc import ABC, abstractmethod
from typing import List, Tuple
from scipy.cluster.hierarchy import linkage, cut_tree
from geo_mapper import GeoMapper
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


def get_mean_th(img: np.ndarray) -> np.ndarray:
    """
    takes in a gray scale image as ndarray and return an threshold based on the first moment of the pixel value
    :param img: gray scale image as ndarray
    :return: binary ndarray, 255 more likely to be panel group part
    """
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def get_sd_th(img: np.ndarray, threshold=2) -> np.ndarray:
    """
    takes in a gray scale image as ndarray and return threshold based on the second order moment of the pixel value
    :param img: gray scale image as ndarray
    :param threshold: threshold value on the standard deviation
    :return: binary ndarray, 255 more likely to be panel group part
    """
    imgf = np.float32(img)
    mu = cv2.blur(imgf, (3, 3))
    mu2 = cv2.blur(imgf * imgf, (3, 3))
    sigma = np.sqrt(mu2 - mu * mu)
    _, th = cv2.threshold(np.uint8(sigma), threshold, 255, cv2.THRESH_BINARY_INV)
    return th


class Panel:
    """this class describes a solar panel on an IR image
    """

    def __init__(self, points: List[List[int]]):
        """
        a panel is assumed to be a rectangle
        :param points: pixel location of top left vertex and bottom right vertex, i.e. [[row1, col1], [row2, col2]],
        counting from 0
        """
        self._p1 = points[0]
        self._p2 = points[1]

    def get_pixel_position(self) -> List[List[int]]:
        return [self._p1, self._p2]

    def get_xy_position(self) -> List[List[int]]:
        x1, y1 = self._p1[1], self._p1[0]
        x2, y2 = self._p2[1], self._p2[0]
        return [[x1, y1], [x2, y2]]


class PanelGroupPart:
    """this class describes a part of panel groups on an ir image
    """
    def __init__(self, contour: np.ndarray):
        """
        :param contour: contour of the part of panel group
        """
        self._cnt = contour
        self._panels = list()

    def get_contour(self) -> np.ndarray:
        return self._cnt

    def add_panel(self, panel: Panel):
        self._panels.append(panel)

    def add_panels(self, panels: List[Panel]):
        self._panels.extend(panels)

    def get_panels(self) -> List[Panel]:
        return self._panels

    def get_centroid(self) -> Tuple[int, int]:
        """
        get the center point pixel position of the panel group part on the image, in form of (x, y)
        :return: centre point of the panel group part
        """
        cnt = self._cnt
        m = cv2.moments(cnt)
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
        return cx, cy


class IRProfile:
    """this is the class describing the profile of an image
    """
    def __init__(self):
        self._panel_groups = list()
        self._img = None

    def set_image(self, img: np.ndarray):
        """
        set the raw image
        :param img: raw gray scale image
        """
        self._img = img

    def add_panel_groups(self, panel_groups: List[PanelGroupPart]):
        self._panel_groups.extend(panel_groups)

    def get_panel_groups(self) -> List[PanelGroupPart]:
        return self._panel_groups

    def draw(self) -> np.ndarray:
        """
        draw the profile on an image
        :return: image with profile drawn on it as ndarray
        """
        img = cv2.cvtColor(self._img, cv2.COLOR_GRAY2BGR)

        if self._panel_groups is not None:
            for panel_group in self.get_panel_groups():
                cv2.drawContours(img, [panel_group.get_contour()], -1, (0, 255, 0), 1)
                cv2.circle(img, panel_group.get_centroid(), 2, (0, 255, 0))
                logger.debug("panel group area is {}".format(cv2.contourArea(panel_group.get_contour())))
                for panel in panel_group.get_panels():
                    cv2.rectangle(img, tuple(panel.get_xy_position()[0]), tuple(panel.get_xy_position()[1]),
                                  (255, 0, 0), 1)
        return img


class IRProfiler(ABC):
    """this is the abstract class for conducting semantic analysis on an infrared image.
    The objective is to output the panels and panel groups
    """

    def __init__(self):
        pass

    @abstractmethod
    def create_profile(self, image_path: str) -> IRProfile:
        """
        trigger the analysis create_profile for an ir image
        :param image_path: the path of an infrared image of gray scale
        :return: image profile
        """
        pass


class IRProfilerTh(IRProfiler):
    """this class does image segmentation based on first order and second order threshold
    """
    def __init__(self, panel_min: int = 100, panel_max: int = 1000, panel_approx_th: int = 5, n_vertex_th: int = 18):
        """
        :param panel_min: minimum area for a panel, in pixels
        :param panel_max: maximum area for a panel, in pixels
        :param panel_approx_th: threshold when approximate a polygon
        :param n_vertex_th: if number of vertices is a=larger than this, it is not considered as a pannel
        """
        IRProfiler.__init__(self)
        self._panel_min = panel_min
        self._panel_max = panel_max
        self._approx_th = panel_approx_th
        self._n_vertex_th = n_vertex_th

    @staticmethod
    def _get_image(image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return img

    def _get_panels(self, img: np.ndarray) -> List[Panel]:
        """
        find the solar panels in an image
        :param img: gray scale image as an ndarray
        :return: list of Panels
        """
        th_mean = get_mean_th(img)
        th_sd = get_sd_th(img)
        th_mean[th_sd == 0] = 0
        #
        cv2.imshow("th", th_mean)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        _, contours, _ = cv2.findContours(th_mean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        panels = list()
        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < self._panel_min or area > self._panel_max:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area

            if solidity < 0.7:
                continue

            # approx = cv2.approxPolyDP(hull, self._approx_th, True)
            approx = cv2.approxPolyDP(cnt, self._approx_th, True)
            if len(approx) > self._n_vertex_th:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            panels.append(Panel([[y, x], [y+h, x+w]]))
        return panels

    def create_profile(self, image_path: str) -> IRProfile:
        image = self._get_image(image_path)
        panels = self._get_panels(image)
        if len(panels) > 0:
            panel_group_parts = [x for x in aggregate_panels(panels) if cv2.contourArea(x.get_contour()) > 2000]
        else:
            panel_group_parts = None
        profile = IRProfile()
        profile.set_image(image)
        profile.add_panel_groups(panel_group_parts)
        return profile


def aggregate_panels(panels: List[Panel]) -> List[PanelGroupPart]:
    """
    the function takes a list of Panels then groups them into a list of PanelGroupParts
    :param panels: a list of Panels
    :return: a list of PanelGroupParts
    """
    vertices = list()
    for p in range(len(panels)):
        panel = panels[p]
        points = panel.get_pixel_position()
        x, y = points[0][1], points[0][0]
        w, h = points[1][1]-x, points[1][0]-y
        vertices.extend([[i, j, p] for i in range(x, x + w, 5) for j in range(y, y + h, 5)])
        vertices.extend([[x, y + h, p], [x + w, y, p], [x + w, y + h, p]])
    linkage_matrix = linkage(np.array(vertices)[:, :2], method="single", metric="chebyshev")
    ctree = cut_tree(linkage_matrix, height=[10])
    cluster = [x[0] for x in ctree]

    panel_groups = list()
    for group in range(max(cluster)+1):
        vertices_group = [vertices[i] for i in range(len(vertices)) if cluster[i] == group]
        contour = cv2.convexHull(np.array([x[:2] for x in vertices_group]))
        panel_index = set([x[2] for x in vertices_group])
        panel_group = PanelGroupPart(contour=contour)
        for i in panel_index:
            panel_group.add_panel(panels[i])
        panel_groups.append(panel_group)
    return panel_groups


def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    import sys
    img_path = sys.argv[1]
    segmentor = IRProfilerTh()
    profile = segmentor.create_profile(img_path)
    img = profile.draw()
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
