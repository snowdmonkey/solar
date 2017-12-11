import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Iterable, Optional, Dict

import torch
import cv2
import json
import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree
from torch.autograd import Variable

# from locate import Station
from detect_hotspot import HotSpotDetector
from geo_mapper import GeoMapper
from fcn.utils import get_panel_group_model

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


class RectProfile:
    """the class describes those profiles that can be represented by a rectangle
    """

    def __init__(self, points: Tuple[Tuple[int, int], Tuple[int, int]]):
        """
        a panel is assumed to be a rectangle
        :param points: pixel location of top left vertex and bottom right vertex, i.e. ((row1, col1), (row2, col2)),
        counting from 0
        """
        self._p1 = points[0]
        self._p2 = points[1]
        self._panel_id = None

    @property
    def points_rc(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        :return: the left-top and bottom-right points in row-column format
        """
        return self._p1, self._p2

    @property
    def points_xy(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        :return: the left-top and bottom-right points in row-column format
        """
        x1, y1 = self._p1[1], self._p1[0]
        x2, y2 = self._p2[1], self._p2[0]
        return (x1, y1), (x2, y2)


class PanelProfile(RectProfile):
    """this class describes a solar panel on an IR image
    """

    def __init__(self, points_rc: Tuple[Tuple[int, int], Tuple[int, int]]):
        """
        a panel is assumed to be a rectangle
        :param points_rc: pixel location of top left vertex and bottom right vertex, i.e. ((row1, col1), (row2, col2)),
        counting from 0
        """
        RectProfile.__init__(self, points_rc)


class DefectProfile(RectProfile):
    """this class describes a defect on an IR image"""

    def __init__(self, points_rc: Tuple[Tuple[int, int], Tuple[int, int]], severity: float):
        """
        a defect profile is assumed to be a rectangle area.
        :param points_rc: the pixel position of the top-left corner and bottom-right corner, in format of (row, col)
        """
        RectProfile.__init__(self, points_rc)

        self._gps = None
        self._utm = None
        self._severity = severity

    @property
    def gps(self) -> Optional[Tuple[float, float]]:
        """
        gps of the top left point, in format (latitude, longitude)
        """
        return self._gps

    @property
    def utm(self) -> Optional[Tuple[float, float, int]]:
        """
        utm of the top left point, in format of (easting, northing, zone_number)
        """
        return self._utm

    @property
    def severity(self) -> float:
        return self._severity

    def set_gps(self, gps: Tuple[float, float]):
        self._gps = gps

    def set_utm(self, utm: Tuple[float, float, int]):
        self._utm = utm


class PanelGroupProfile:
    """this class describes a part of panel groups on an ir image
    """

    def __init__(self, contour: np.ndarray):
        """
        :param contour: contour of the part of panel group
        """
        self._cnt = contour
        self._panels = list()
        self._panel_group_id = None
        self._defects = list()
        self._bounds = None  # x, y, w, h
        self._get_bounds()

    def _get_bounds(self):
        """
        set the bounds of the panel group, in format of x, y, w, h
        """
        self._bounds = cv2.boundingRect(self._cnt)

    @property
    def most_left(self) -> int:
        return self._bounds[0]

    @property
    def most_right(self) -> int:
        return self._bounds[0] + self._bounds[2]

    @property
    def most_top(self) -> int:
        return self._bounds[1]

    @property
    def most_bottom(self) -> int:
        return self._bounds[1] + self._bounds[3]

    @property
    def contour(self) -> np.ndarray:
        return self._cnt

    def add_panel(self, panel: PanelProfile):
        self._panels.append(panel)

    def add_panels(self, panels: List[PanelProfile]):
        self._panels.extend(panels)

    @property
    def defects(self) -> List[DefectProfile]:
        return self._defects

    def add_defect(self, defect: DefectProfile):
        self._defects.append(defect)

    @property
    def panel_group_id(self) -> str:
        return self._panel_group_id

    def set_panel_group_id(self, group_id):
        self._panel_group_id = group_id

    @property
    def panels(self) -> List[PanelProfile]:
        return self._panels

    @property
    def centroid_xy(self) -> Tuple[int, int]:
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
    """this is the class describing the profile of a gray scale ir image
    """

    def __init__(self):
        self._panel_groups = list()
        self._img = None
        self._geo_mapper = None
        self._height = None
        self._width = None
        self._name = None

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def image(self) -> Optional[np.ndarray]:
        return self._img

    @property
    def geo_mapper(self) -> Optional[GeoMapper]:
        return self._geo_mapper

    def set_name(self, name: str):
        self._name = name

    def set_image(self, img: np.ndarray):
        """
        set the raw image
        :param img: raw gray scale image
        """
        self._img = img
        self._height, self._width = img.shape

    def set_geo_mapper(self, mapper: GeoMapper):
        """
        set the geo mapper of this profile
        :param mapper:
        """
        self._geo_mapper = mapper

    def add_panel_groups(self, panel_groups: Iterable[PanelGroupProfile]):
        self._panel_groups.extend(panel_groups)

    @property
    def panel_groups(self) -> List[PanelGroupProfile]:
        return self._panel_groups

    def draw(self) -> np.ndarray:
        """
        draw the profile on an image
        :return: image with profile drawn on it as ndarray
        """
        img = cv2.cvtColor(self._img, cv2.COLOR_GRAY2BGR)

        if self._panel_groups is not None:
            for panel_group in self.panel_groups:
                cv2.drawContours(img, [panel_group.contour], -1, (0, 255, 0), 1)
                # cv2.circle(img, panel_group.centroid_xy, 2, (0, 255, 0))
                cv2.putText(img, panel_group.panel_group_id, panel_group.centroid_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1, cv2.LINE_AA)
                # cv2.putText(img, "test", panel_group.centroid_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                # (255, 255, 0), 1, cv2.LINE_AA)
                logger.debug("panel group area is {}".format(cv2.contourArea(panel_group.contour)))
                for panel in panel_group.panels:
                    cv2.rectangle(img, tuple(panel.points_xy[0]), tuple(panel.points_xy[1]),
                                  (255, 0, 0), 1)
                for defect in panel_group.defects:
                    cv2.rectangle(img, defect.points_xy[0], defect.points_xy[1], (0, 0, 255))
        return img

    def to_dict(self) -> Dict:
        """
        :return: a dict contain the information of the profile
        """
        d = {"name": self.name, "panelGroups": list()}
        for panel_group in self._panel_groups:
            panel_group_dict = {"centroidXy": list(panel_group.centroid_xy),
                                "groupId": panel_group.panel_group_id, "rects": list()}
            for defect in panel_group.defects:
                defect_dict = {"leftTop": list(defect.points_xy[0]), "rigthBottom": list(defect.points_xy[1])}
                panel_group_dict.get("rects").append(defect_dict)
            d.get("panelGroups").append(panel_group_dict)
        return d

    def to_json(self) -> str:
        """
        :return: a json string contain the information of the profile
        """
        return json.dumps(self.to_dict())


class IRProfiler(ABC):
    """this is the abstract class for conducting semantic analysis on an infrared image.
    The objective is to output the panels and panel groups
    """

    def __init__(self, panel_group_min: int = 2000):
        """
        :param panel_group_min: the minimum area of a panel group, in pixels
        """
        self._panel_group_min = panel_group_min

    @abstractmethod
    def create_profile(self, image_path: str) -> IRProfile:
        """
        trigger the analysis create_profile for an ir image
        :param image_path: the path of an infrared image of gray scale
        :return: image profile
        """
        pass

    @staticmethod
    def _get_image(image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return img

    @staticmethod
    def mark_defects(image: np.ndarray, panel_group: PanelGroupProfile):
        """
        this method aim to generate defect profiles for a panel group profile and attach them to it
        :param image: group scale image as numpy array, it serve as the background for the profiling
        :param panel_group: the panel group profile
        """
        sub_img = np.zeros_like(image)
        cv2.drawContours(sub_img, [panel_group.contour], -1, 255, -1)
        sub_img[sub_img == 255] = image[sub_img == 255]
        hot_spot_detector = HotSpotDetector(sub_img, 4.0)
        hot_spot = hot_spot_detector.get_hot_spot()
        points = hot_spot.points

        mask = np.zeros_like(image)
        for point in points:
            mask[tuple(point)] = 255

        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.convexHull(x) for x in contours if cv2.contourArea(x) > 5]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            defect_mean = cv2.mean(image, mask)[0]
            severity = (defect_mean-hot_spot.bg_mean) / hot_spot.bg_sd

            defect = DefectProfile(points_rc=((y, x), (y + h, x + w)), severity=severity)
            panel_group.add_defect(defect)


class ThIRProfiler(IRProfiler):
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

    def _get_panels(self, img: np.ndarray) -> List[PanelProfile]:
        """
        find the solar panels in an image
        :param img: gray scale image as an ndarray
        :return: list of Panels
        """
        th_mean = get_mean_th(img)
        th_sd = get_sd_th(img)
        th_mean[th_sd == 0] = 0
        #
        # cv2.imshow("th", th_mean)
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
            panels.append(PanelProfile(((y, x), (y + h, x + w))))
        return panels

    def create_profile(self, image_path: str) -> IRProfile:
        image = self._get_image(image_path)
        panels = self._get_panels(image)
        if len(panels) > 0:
            panel_group_parts = [x for x in aggregate_panels(panels)
                                 if cv2.contourArea(x.contour) > self._panel_group_min]
        else:
            panel_group_parts = list()
        profile = IRProfile()
        profile.set_image(image)
        profile.add_panel_groups(panel_group_parts)
        for group in profile.panel_groups:
            self.mark_defects(image, group)
        return profile


def aggregate_panels(panels: List[PanelProfile]) -> List[PanelGroupProfile]:
    """
    the function takes a list of Panels then groups them into a list of PanelGroupParts
    :param panels: a list of Panels
    :return: a list of PanelGroupParts
    """
    vertices = list()
    for p in range(len(panels)):
        panel = panels[p]
        points = panel.points_rc
        x, y = points[0][1], points[0][0]
        w, h = points[1][1] - x, points[1][0] - y
        vertices.extend([[i, j, p] for i in range(x, x + w, 5) for j in range(y, y + h, 5)])
        vertices.extend([[x, y + h, p], [x + w, y, p], [x + w, y + h, p]])
    linkage_matrix = linkage(np.array(vertices)[:, :2], method="single", metric="chebyshev")
    ctree = cut_tree(linkage_matrix, height=[10])
    cluster = [x[0] for x in ctree]

    panel_groups = list()
    for group in range(max(cluster) + 1):
        vertices_group = [vertices[i] for i in range(len(vertices)) if cluster[i] == group]
        contour = cv2.convexHull(np.array([x[:2] for x in vertices_group]))
        panel_index = set([x[2] for x in vertices_group])
        panel_group = PanelGroupProfile(contour=contour)
        for i in panel_index:
            panel_group.add_panel(panels[i])
        panel_groups.append(panel_group)
    return panel_groups


class FcnIRProfiler(IRProfiler):

    def __init__(self):
        super().__init__()

    def _get_panel_groups(self, image: np.ndarray) -> List[PanelGroupProfile]:

        model = get_panel_group_model()
        model.eval()
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        tensor = torch.from_numpy(image / 255)
        data = Variable(tensor, volatile=True).float()
        output = model(data)
        bs, c, h, w = output.size()
        _, indices = output.data.max(1)
        indices = indices.view(bs, h, w)
        output = indices.numpy()[0]
        th = np.uint8(output * 255)

        _, contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [PanelGroupProfile(cnt) for cnt in contours if cv2.contourArea(cnt) > self._panel_group_min]

    def create_profile(self, image_path: str) -> IRProfile:
        image = self._get_image(image_path)
        panel_groups = self._get_panel_groups(image)
        profile = IRProfile()
        profile.set_image(image)
        profile.add_panel_groups(panel_groups)
        for group in profile.panel_groups:
            self.mark_defects(image, group)
        return profile


def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("img", type=str, help="path of an IR image")
    args = parser.parse_args()

    # segmentor = ThIRProfiler()
    segmentor = FcnIRProfiler()
    profile = segmentor.create_profile(args.img)
    img = profile.draw()
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
