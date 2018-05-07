from abc import ABC, abstractmethod
from typing import Tuple, Dict, List
from xml.dom import minidom
from geo_mapper import GeoMapper, TifGeoMapper
import numpy as np
import cv2
import argparse
import os
import json


class PanelGroupLabelResult:
    """
    this class serves as the results for panel group labeler
    """

    def __init__(self, groups: Dict[str, List[Tuple[int, int]]], width: int, height: int):
        """
        constructor
        :param groups: a dict group_id -> [(conner1_row, corner1_col), (conner2_row, corner2_col),
        (conner3_row, corner3_col), (conner4_row, corner4_col),...]. The order of the vertices should be clockwise or
        counter clockwise
        :param width: width of the original panorama in pixels
        :param height: height of the original panorama in pixels
        """
        self._groups = groups
        self._width = width
        self._height = height

    @property
    def groups(self) -> Dict[str, List[Tuple[int, int]]]:
        return self._groups

    def dump_svg(self, path: str = "panelGroups.svg"):
        """
        transfer the panel groups location into a svg file, so it can be loaded into Gimp for manual adjustment
        :return: None
        """
        with open(path, "w") as f:
            f.write('<svg width="' + str(self._width) + '" height="' + str(self._height) +
                    '" xmlns="http://www.w3.org/2000/svg">')
            f.write("\n")
            for poly in self._groups.values():
                f.write("<polygon points=\"")
                for row, col in poly:
                    f.write("{},{} ".format(col, row))
                f.write("\"/>\n")
            f.write("</svg>")

    def dump_json(self, mapper: GeoMapper, path: str = "groupPanel.json"):
        """
        dump the label results to a json file that will eventually be used at backend
        :param path: path to the json to dump the label results
        :param mapper: geo mapper that can map the pixel location to gps coordinates
        :return: None
        """
        groups = self._groups
        results = list()
        for group_id, vertices in groups.items():
            vertices_gps = [mapper.pixel2gps(row=x[0], col=x[1]) for x in vertices]
            d = {"vertices": vertices_gps, "groupId": group_id}
            results.append(d)

        results.sort(key=lambda x: x.get("groupId"))

        with open(path, "w") as f:
            json.dump(results, f)


class PanelGroupLabeler(ABC):

    @staticmethod
    def _group_to_order(group: List[Tuple[int, int]]) -> float:
        row_nums = [x[0] for x in group]
        col_nums = [x[1] for x in group]
        row_avg = sum(row_nums)/len(row_nums)
        col_avg = sum(col_nums)/len(col_nums)
        return 10*row_avg+col_avg

    @staticmethod
    def _code_groups(group_list: List[List[Tuple[int, int]]]) -> Dict[str, List[Tuple[int, int]]]:
        """
        give each group an Id
        """
        d = dict()
        group_list.sort(key=PanelGroupLabeler._group_to_order)
        for i, group in enumerate(group_list):
            d.update({"G{:05}".format(i): group})
        return d

    @abstractmethod
    def _process_image(self, img_path: str) -> List[List[Tuple[int, int]]]:
        """
        create_profile a given image
        :param img_path:
        :return: list of groups, groups = [group], group = [(conner1_row, corner1_col), (conner2_row, corner2_col),
        (conner3_row, corner3_col), (conner4_row, corner4_col),...]. The vertices should be sorted clockwise or
        counter clockwise
        """
        pass

    def label(self, img_path: str) -> PanelGroupLabelResult:
        group_list = self._process_image(img_path)
        coded_groups = self._code_groups(group_list)
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        return PanelGroupLabelResult(groups=coded_groups, width=width, height=height)

    @staticmethod
    def _str2coordinates(s: str) -> Tuple[int, int]:
        words = s.split(",")
        words = [int(float(x)) for x in words]
        return words[1], words[0]

    def from_svg(self, svg_path: str) -> PanelGroupLabelResult:
        """
        generate panel group result from a svg file. The svg file show be an exported path from GIMP
        :param svg_path: path to the svg file
        :return: panel group label result
        """
        doc = minidom.parse(svg_path)
        path_string = doc.getElementsByTagName("path")[0].getAttribute("d")  # type: str
        view_str = doc.getElementsByTagName("svg")[0].getAttribute("viewBox")  # type: str

        doc.unlink()

        words = path_string.split()

        groups = list()
        group_str = list()
        collecting = False

        for word in words:
            if word == "C":
                group_str = list()
                collecting = True
            elif word == "Z":
                collecting = False
                group_str = [s for i, s in enumerate(group_str) if i % 3 == 0]
                group = [self._str2coordinates(x) for x in group_str]
                groups.append(group)
            elif collecting is True:
                group_str.append(word)

        words = [int(x) for x in view_str.split(" ")]
        width = words[2]
        height = words[3]

        groups = self._code_groups(groups)

        return PanelGroupLabelResult(groups=groups, width=width, height=height)


# class SolarPanelReco(PanelGroupLabeler):
#
#     img_path = ''
#     img_roi = ''
#     img_roi_black = ''
#     boundary_value = 0 #4540
#     host_no = [] # the NO. of group string solar panel
#     sub_no = []  # the NO. of single solar panel
#
#     def __init__(self, img_path: str, img_roi: str, img_roi_black: str, boundary_value):
#         self.img_path = img_path
#         self.img_roi = img_roi
#         if boundary_value > 0:
#             self.img_roi_black = img_roi_black
#             self.boundary_value = boundary_value
#
#
#     def process_image(self) -> Dict[str, List[Tuple[int, int]]]:
#         """
#         process a given image
#         :param img_path:
#         :return: a dict group_id -> [(conner1_row, corner1_col), (conner2_row, corner2_col), (conner3_row, corner3_col),
#          (conner4_row, corner4_col)]
#         """
#         #results = dict({'name':[(4200,581),(4486,581),(4486,617),(4200,617)]})
#
#
#         results = dict()
#         results1 = dict()
#         results_group = dict()
#         img_rgb = cv2.imread(self.img_path)
#         img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#
#         template = cv2.imread(self.img_roi,0)
#         w, h = template.shape[::-1]
#         res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
#         threshold = 0.7
#         loc = np.where( res >= threshold)
#
#         template_black = cv2.imread(self.img_roi_black,0)
#         w1, h1 = template_black.shape[::-1]
#         res_black = cv2.matchTemplate(img_gray,template_black,cv2.TM_CCOEFF_NORMED)
#         loc_black = np.where( res_black >= threshold)
#
#         loc_modify = self._postProcess2(self._postProcess1(loc[::-1]), h)
#         loc_modify = [a for a in loc_modify if a[1] < self.boundary_value]
#
#         loc_modify_black = self._postProcess2(self._postProcess1(loc_black[::-1]), h1)
#         loc_modify_black = [a for a in loc_modify_black if a[1] > self.boundary_value]
#
#         loc_modify = self._verify(loc_modify)
#         loc_modify_black = self._verify(loc_modify_black)
#
#         host_loc = self.postProcess3(loc_modify, w, h) + self.postProcess3(loc_modify_black, w1, h1)
#
#         #tmp_len = len(loc_modify)
#         print(len(loc_modify),len(loc_modify_black), len(self.sub_no))
#         for i, pt in enumerate(loc_modify):
#             results.update({self.sub_no[i]:[(pt[0], pt[1]), (pt[0] + w, pt[1]), (pt[0] + w, pt[1] + h), (pt[0], pt[1] + h)]})
#
#         for i, pt in enumerate(loc_modify_black):
#             results1.update({self.sub_no[i]:[(pt[0], pt[1]), (pt[0] + w1, pt[1]), (pt[0] + w1, pt[1] + h1), (pt[0], pt[1] + h1)]})
#
#         for i in range(len(host_loc)-1):
#             if i % 2 == 0:
#                 results_group.update({self.host_no[i//2]:[(host_loc[i][0], host_loc[i][1]), (host_loc[i+1][0], host_loc[i][1]), (host_loc[i+1][0], host_loc[i+1][1]), (host_loc[i][0], host_loc[i+1][1])]})
#
#         # count = 0
#         # for pt in zip(*loc[::-1]):
#             # #print(pt[0],pt[1])
#             # count +=1
#             # results.update({str(count):[(pt[0], pt[1]), (pt[0] + w, pt[1]), (pt[0] + w, pt[1] + h), (pt[0], pt[1] + h)]})
#
#         return results, results1, results_group
#
#     @staticmethod
#     def _postProcess1(location = np.array([])):
#         '''
#         delete the duplicate coordinate
#         param: loc of solar panel
#         return: loc without duplicate coordinate
#         '''
#         b = list(set(location[1]))
#         b.sort()
#         loc = [[], []]
#         for item in b:
#             a = []
#             for y in range(len(location[1])):
#                 if item == (location[1][y]):
#                     a.append(location[0][y])
#             for i in range(len(a)-1, -1, -1):
#                 if a[i] == a[i-1] + 1:
#                     a.pop(i)
#             for item_x in a:
#                 loc[0].append(item_x)
#                 loc[1].append(item)
#         return loc
#
#     @staticmethod
#     def _postProcess2(location, h):
#         '''
#         delete the duplicate coordinate
#         param: loc of solar panel
#         return: loc without duplicate coordinate
#         '''
#         arr = np.array(location)
#         list = arr.transpose().tolist()
#         list.sort()
#         for i in range(len(list)-1, -1, -1):
#             if (list[i][0] - list[i-1][0] == 1 or list[i][0] - list[i-1][0] == 0):
#                 if abs(list[i][1] - list[i-1][1]) <= 5 or abs(list[i][1] - list[i-1][1]) == h/2:
#                     list.pop(i)
#
#         for j in range(len(list)-1, -1, -1):
#             if ([list[j][0] - 1, list[j][1] + 1] in list) or ([list[j][0] - 1, list[j][1]] in list) or ([list[j][0] - 1, list[j][1] - 1] in list) or ([list[j][0] - 2, list[j][1] + 1] in list) or ([list[j][0] - 2, list[j][1] - 1] in list):
#                 list.pop(j)
#
#         for k in range(len(list)-1, -1, -1):
#             if ([list[k][0] - 1, list[k][1] + 2] in list) or ([list[k][0] - 1, list[k][1] - 2] in list) or ([list[k][0] - 1, list[k][1] + 3] in list) or ([list[k][0] - 1, list[k][1] - 3] in list) or ([list[k][0] - 1, list[k][1] + 4] in list) or ([list[k][0] - 1, list[k][1] - 4] in list) or ([list[k][0] - 1, list[k][1] + 5] in list) or ([list[k][0] - 1, list[k][1] - 5] in list) or ([list[k][0] - 2, list[k][1] + 2] in list) or ([list[k][0] - 2, list[k][1]] in list) or ([list[k][0] - 2, list[k][1] - 2] in list) or ([list[k][0] - 2, list[k][1] + 3] in list) or ([list[k][0] - 2, list[k][1] - 3] in list):
#                 list.pop(k)
#
#         for i in range(len(list)-1, -1, -1):
#             if (list[i][0] - list[i-1][0] == 1 or list[i][0] - list[i-1][0] == 0) and abs(list[i][1] - list[i-1][1]) == h/2:
#                 list.pop(i)
#
#         list.sort(key=lambda x:x[1])
#         return list
#
#     def postProcess3(self, location, w ,h):
#         '''
#         output the group coordinate
#         param: loc of solar panel, width & height of solar panel
#         return: pixel cordinate list of group solar panel
#         '''
#         host = []
#         m = 0
#         count = len(self.host_no)     # count the group number of solar panel
#         count_sub = 0 # count the single solar panel
#         for i in range(len(location)-1):
#             tmp = []
#             if location[i+1][1] - location[i][1] > 19:#100, the parameter 100 or 70 need to be set
#                 n = m
#                 m = i+1
#                 tmp = location[n:m] # get a row of solar panel
#                 tmp.sort()
#                 host.append(tmp[0])
#                 count = count + 1
#                 #count = (len(host) + 1)//2
#                 self.host_no.append('G' + '%05d' % count)
#                 for j in range(len(tmp)-1):
#                     count_sub = count_sub + 1
#                     if tmp[j+1][0] - tmp[j][0] > 16: #32 # the parameter 16 or 32 need to be set, w+2 is recomended
#                         for no in range(count_sub):
#                             self.sub_no.append(self.host_no[-1] + 'S' + '%02d' % (no+1) + '\n')
#                         count_sub = 0
#                         host.append([tmp[j][0]+ w,tmp[j][1]+ h])
#                         host.append(tmp[j+1])
#                         count = count + 1
#                         self.host_no.append('G' + '%05d' % count)
#                     if j == len(tmp)-2:
#                         for no in range(count_sub):
#                             self.sub_no.append(self.host_no[-1] + 'S' + '%02d' % (no+1) + '\n')
#                         self.sub_no.append(self.host_no[-1] + 'S' + '%02d' % (count_sub + 1) + '\n')
#                         count_sub = 0
#                         host.append([tmp[j+1][0]+ w, tmp[j+1][1]+ h])
#
#             if i == len(location)-2:
#                 tmp = location[m:len(location)] # get the last row of solar panel
#                 tmp.sort()
#                 host.append(tmp[0])
#                 count = count + 1
#                 self.host_no.append('G' + '%05d' % count)
#                 for j in range(len(tmp)-1):
#                     count_sub = count_sub + 1
#                     if tmp[j+1][0] - tmp[j][0] > 16:
#                         for no in range(count_sub):
#                             self.sub_no.append(self.host_no[-1] + 'S' + '%02d' % (no+1) + '\n')
#                         count_sub = 0
#                         host.append([tmp[j][0]+ w,tmp[j][1]+ h])
#                         host.append(tmp[j+1])
#                         count = count + 1
#                         self.host_no.append('G' + '%05d' % count)
#                     if j == len(tmp)-2:
#                         for no in range(count_sub):
#                             self.sub_no.append(self.host_no[-1] + 'S' + '%02d' % (no+1) + '\n')
#                         self.sub_no.append(self.host_no[-1] + 'S' + '%02d' % (count_sub + 1) + '\n')
#                         count_sub = 0
#                         host.append([tmp[j+1][0]+ w, tmp[j+1][1]+ h])
#         return host
#
#     @staticmethod
#     def _verify(list):
#         """
#         for verify the solar panel which was not recognised correctly
#         this function is not must
#         """
#         temp_del = [[1420,4509], [4845,803], [5650,1526], [4327,2144], [4929,2139], [4255,4666], [4464,4868], [2577,5134], [1785,5029], [2293,5388], [1918,5480], [1721,5478], [2376,7210], [1906,8235], [3417,5890], [3857,5054], [4968,4301]]
#         temp_add = []
#         for item in temp_del:
#             try:
#                 list.remove(item)
#             except ValueError:
#                 continue
#         return list


class ColorBasedLabeler(PanelGroupLabeler):

    def _process_image(self, img_path: str) -> List[List[Tuple[int, int]]]:
        """
        implement abstract method in PanelGroupLabeler
        """
        results = list()
        raw_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        blue_scale = self._convert_blue_scale(raw_image)
        _, th = cv2.threshold(blue_scale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

        _, contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [x for x in contours if cv2.contourArea(x) > 2500]
        for cnt in contours:
            # x, y, w, h = cv2.boundingRect(cnt)
            # results.append([(y, x), (y+h, x), (y+h, x+w), (y, x+w)])
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = box.astype(np.uint)
            box = box.tolist()
            results.append([tuple(x[::-1]) for x in box])
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
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="sub_command", help="sub-command help")

    parser_tosvg = subparsers.add_parser("tosvg", help="write panel group profile into a svg file")
    parser_tosvg.add_argument("image_path", type=str, help="path to panorama image")
    parser_tosvg.add_argument("--out-dir", type=str, default=".", required=False)

    parser_fromsvg = subparsers.add_parser("fromsvg", help="generate panelGroup.json from a svg file")
    parser_fromsvg.add_argument("svg_file", type=str, help="path to the svg file")
    parser_fromsvg.add_argument("tif_file", type=str, help="path to the geo tif file")
    parser_fromsvg.add_argument("--out-dir", type=str, default=".", required=False)

    args = parser.parse_args()

    labeler = ColorBasedLabeler()

    if args.sub_command == "tosvg":
        image_path = args.image_path
        out_dir = args.out_dir
        # labeler = ColorBasedLabeler()
        result = labeler.label(image_path)
        result.dump_svg(os.path.join(out_dir, "panelgroup_auto.svg"))
    elif args.sub_command == "fromsvg":
        svg_path = args.svg_file
        tif_path = args.tif_file
        out_dir = args.out_dir
        result = labeler.from_svg(svg_path)
        result.dump_json(mapper=TifGeoMapper(tif_path), path=os.path.join(out_dir, "groupPanel.json"))


# def myMain():
#     img_path = sys.argv[1]
#     img_roi = sys.argv[2]
#     img_roi_black = sys.argv[3]
#     #boundary_value = sys.argv[4]
#     labeler = SolarPanelReco(img_path, img_roi, img_roi_black, 4540)
#     result,result1, result_group = labeler.process_image()
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#
#     for k, v in result.items():
#         cv2.rectangle(img, (v[0][0], v[0][1]), (v[2][0], v[2][1]), (0, 0, 255), 1)
#
#     for k, v in result_group.items():
#         cv2.rectangle(img, (v[0][0], v[0][1]), (v[2][0], v[2][1]), (0, 0, 0), 2)
#
#     cv2.imwrite("mylabeled.png", img)


if __name__ == "__main__":
    main()
    # myMain()