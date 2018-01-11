# import exifread
import base64
import json
import logging
import os
import subprocess
from os import listdir
from os.path import join, basename
from typing import Union, List, Dict

import cv2
import numpy as np
import utm
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, cut_tree
from defect_category import DefectCategory
from detect_hotspot import HotSpotDetector
from extract_rect import rotate_and_scale, PanelCropper
from geo_mapper import UTMGeoMapper
from locate import Station, Positioner, PanelGroup
from semantic import FcnIRProfiler

plt.switch_backend("agg")

logger = logging.getLogger(__name__)


def _get_raw_from_string(s: str, depth: int = 16) -> np.ndarray:
    """
    return an image from a base64 encoded string
    :param s: base64 encoded string
    :param depth: depth of the image sample
    :return: image as ndarray
    """
    data = base64.b64decode(s)
    if depth == 16:
        dtype = np.uint16
    elif depth == 8:
        dtype = np.uint8
    else:
        raise ValueError

    img = cv2.imdecode(np.frombuffer(data, dtype=dtype), cv2.IMREAD_ANYDEPTH)
    return img


def batch_process_exif(folder_path: str, outfile_path=None) -> List[Dict]:
    """
    use exiftool to extract exif information, include the camera's gps, relative altitude, gesture
    :param folder_path: the path of folder that contains the IR images
    :param outfile_path: the path of json file to store the exif information, default to be under folder_path
    :return: a dictionary with key as the image name and value as the exif information
    """

    if outfile_path is None:
        outfile_path = join(folder_path, "exif.json")

    file_names = [x for x in os.listdir(folder_path) if x.endswith(".jpg")]

    cmd = ['exiftool', "-j", "-b", "-c", "%+.10f"]
    # results = list()
    for file_name in file_names:
        cmd.append(join(folder_path, file_name))

    # logger.info("start to extract exif from {}".format(file_name))

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    out = proc.stdout
    results = json.loads(out.decode("utf-8"))
        # results.append(result)
    for result in results:
        result["GPSLatitude"] = float(result.get("GPSLatitude"))
        result["GPSLongitude"] = float(result.get("GPSLongitude"))
        base_name = result.get("FileName").replace(".jpg", "")
        result["image"] = base_name

    with open(outfile_path, "w") as outfile:
        json.dump(results, outfile)

    return results


def batch_process_rotation(folder_path: str, exif_path: Union[None, str] = None):
    """
    the function will create a sub folder under folder_path which contains the rotated images. And the images under
    folder path will be rotated so they will all head north. And the degrees for rotate each image should be provided
    in a json file at exif_path with key "GimbalYawDegree"
    :param folder_path: path of the folder contains raw images
    :param exif_path: path of the json file contain exif information, defaults to be under folder_path
    :return: nothing
    """

    rotate_folder_path = join(folder_path, "rotated")
    rotate_raw_folder_path = join(folder_path, "rotated-raw")
    if not os.path.exists(rotate_folder_path):
        os.mkdir(rotate_folder_path)
    if not os.path.exists(rotate_raw_folder_path):
        os.mkdir(rotate_raw_folder_path)
    if exif_path is None:
        exif_path = join(folder_path, "exif.json")
    with open(exif_path) as file:
        exif = json.load(file)
    # file_names = [x for x in listdir(folder_path) if x.endswith(".jpg")]

    for d in exif:
        file_name = d.get("FileName")
        image_path = join(folder_path, file_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        raw = _get_raw_from_string(d.get("RawThermalImage")[7:])
        degree = d.get("GimbalYawDegree")
        if degree is not None:
            if abs(degree) < 10:
                rotated_img = img
                rotated_raw = raw
            elif abs(degree - 90.0) < 10:
                rotated_img = rotate_and_scale(img, -90.0)
                rotated_raw = rotate_and_scale(raw, -90.0)
            elif abs(degree + 90.0) < 10:
                rotated_img = rotate_and_scale(img, 90.0)
                rotated_raw = rotate_and_scale(raw, 90.0)
            elif abs(degree - 180.0) < 10:
                rotated_img = rotate_and_scale(img, 180.0)
                rotated_raw = rotate_and_scale(raw, 180.0)
            elif abs(degree + 180.0) < 10:
                rotated_img = rotate_and_scale(img, 180.0)
                rotated_raw = rotate_and_scale(raw, 180.0)
            else:
                logging.warning("%s is ignored since its yaw is %f degrees", file_name, degree)
                continue
            rotated_img_path = join(rotate_folder_path, file_name)
            rotated_raw_path = join(rotate_raw_folder_path, file_name)
            cv2.imwrite(rotated_img_path, rotated_img)
            cv2.imwrite(rotated_raw_path.replace(".jpg", ".tif"), rotated_raw)


def batch_process_label(folder_path: str) -> List[dict]:
    """
    label the images under the rotated sub-directory of folder_path, label the defects with red rectangle
    :param folder_path: folder for the raw images
    :return: list of image semantic analysis results
    """
    rotate_folder_path = join(folder_path, "rotated")
    label_folder_path = join(folder_path, "labeled")
    results = list()
    if not os.path.exists(label_folder_path):
        os.mkdir(label_folder_path)
    file_names = [x for x in listdir(rotate_folder_path) if x.endswith(".jpg")]
    for file_name in file_names:
        logger.info("labeling file %s", file_name)
        base_name = os.path.splitext(basename(file_name))[0]
        file_path = join(rotate_folder_path, file_name)
        panel_cropper = PanelCropper(file_path)
        # sub_imgs = panel_cropper.get_sub_imgs(rotate_n_crop=False, min_area=5000, max_area=21500,
        #                                       verify_rectangle=10, n_vertices_threshold=8)
        sub_imgs = panel_cropper.get_panels(min_area=100, max_area=1000, n_vertices_threshold=6, approx_threshold=2,
                                            min_panel_group_area=2000)
        if len(sub_imgs) == 0:
            continue

        else:
            points = list()
            for img in sub_imgs:
                hot_spot_detector = HotSpotDetector(img, 4.0)
                hot_spot = hot_spot_detector.get_hot_spot()
                points.extend(hot_spot.points)

            mask = np.zeros_like(panel_cropper.raw_img)
            for point in points:
                mask[tuple(point)] = 255

            _, contours, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cv2.convexHull(x) for x in contours if cv2.contourArea(x) > 5]

            if len(contours) == 0:
                continue
            raw_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            rectangles = list()
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h < 200 * w:
                    rectangles.append((x, y, w, h))
            if len(rectangles) > 0:
                result = dict()
                result.update({"image": base_name,
                               "rects": list(),
                               "height": raw_image.shape[0],
                               "width": raw_image.shape[1]})

                for rectangle in rectangles:
                    x, y, w, h = rectangle
                    cv2.rectangle(raw_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    result.get("rects").append({"x": x, "y": y, "w": w, "h": h})

                results.append(result)
                labeled_img_path = join(label_folder_path, file_name)
                cv2.imwrite(labeled_img_path, raw_image)
    outfile_path = join(folder_path, "rect.json")
    with open(outfile_path, "w") as file:
        json.dump(results, file)

    return results


def batch_process_profile(folder_path: str, gsd: float) -> List[dict]:
    """
    create profile for the images under the rotated sub-directory of folder path
    :param folder_path: folder for the raw image
    :param gsd: ground sampling distance, in meters
    :return: list of image semantic analysis results
    """
    rotated_folder_path = join(folder_path, "rotated")
    profile_folder_path = join(folder_path, "profile")
    affine_folder_path = join(folder_path, "affine")

    if not os.path.exists(profile_folder_path):
        os.mkdir(profile_folder_path)

    if not os.path.exists(affine_folder_path):
        os.mkdir(affine_folder_path)

    # profiler = ThIRProfiler()
    profiler = FcnIRProfiler()
    positioner = Positioner()
    station = Station()

    with open(join(folder_path, "exif.json"), "r") as f:
        exif = json.load(f)

    with open(join(folder_path, "..", "..", "groupPanel.json"), "r") as f:
        group_locations = json.load(f)

    for d in group_locations:
        panel_group = PanelGroup(group_id=d.get("groupId"), vertices_gps=[tuple(x) for x in d.get("vertices")])
        station.add_panel_group(panel_group)

    results = list()
    profile_results = list()

    for d in exif:
        base_name = d.get("image")

        logger.info("start to analysis image {}".format(base_name))

        image_path = join(rotated_folder_path, d.get("FileName"))

        if not os.path.exists(image_path):
            continue

        profile = profiler.create_profile(image_path)

        # construct the geo mapper
        image_latitude = d.get("GPSLatitude")
        image_longitude = d.get("GPSLongitude")
        # image_height = d.get("ImageWidth")
        # image_width = d.get("ImageHeight")
        image_height = profile.height
        image_width = profile.width

        geo_mapper = UTMGeoMapper(gsd=gsd, origin_gps=(image_latitude, image_longitude),
                                  origin_pixel=(image_height / 2 - 0.5, image_width / 2 - 0.5))

        # positioning
        matrix = positioner.locate(profile, geo_mapper, station)

        # save affine transformation figure for checking
        if matrix is not None:
            fig = plt.figure()
            positioner.draw_calibration(profile, geo_mapper, station, matrix, fig)
            fig.savefig(join(affine_folder_path, "{}.jpg".format(base_name)))
            plt.close(fig)

        # save the profile to image for checking
        cv2.imwrite(join(profile_folder_path, "{}.jpg".format(base_name)), profile.draw())

        # record the analysis results
        rects = list()
        for panel_group in profile.panel_groups:
            for defect in panel_group.defects:
                rect = {"x": defect.points_xy[0][0],
                        "y": defect.points_xy[0][1],
                        "w": defect.points_xy[1][0] - defect.points_xy[0][0],
                        "h": defect.points_xy[1][1] - defect.points_xy[0][1],
                        "easting": defect.utm[0],
                        "northing": defect.utm[1],
                        "utm_zone": defect.utm[2],
                        "panel_group_id": panel_group.panel_group_id,
                        "severity": defect.severity}
                rects.append(rect)
        if len(rects) > 0:
            result = {"image": base_name, "height": image_height, "width": image_width, "rects": rects}
            results.append(result)
        profile_results.append(profile.to_dict())

    outfile_path = join(folder_path, "rect.json")
    with open(outfile_path, "w") as file:
        json.dump(results, file)

    with open(join(folder_path, "profile.json"), "w") as f:
        json.dump(profile_results, f)

    return results


def batch_process_aggregate(folder_path: str, group_criteria: float) -> List[dict]:
    """
    this function will read all the labeled defects from ./rect.json, aggregate close ones and output the aggregated
    defects information to ./defects.json
    :param folder_path: folder_path of the raw ir images
    :param group_criteria: in meters, if two defects are closer than this, they will be aggregated
    :return: list of information about defects
    """
    with open(join(folder_path, "exif.json"), "r") as f:
        exif = json.load(f)

    with open(join(folder_path, "rect.json"), "r") as f:
        rect_info = json.load(f)

    rects = list()
    for d in rect_info:
        for rect in d.get("rects"):
            rect.update({"height": d.get("height"),
                         "width": d.get("width"),
                         "image": d.get("image")})
            rects.append(rect)

    group_ids = set([x.get("panel_group_id") for x in rects])

    defect_num = 0
    defects = list()

    for group_id in group_ids:
        rects_match_id = [x for x in rects if x.get("panel_group_id") == group_id]

        if len(rects_match_id) == 1:
            cluster = [0]
        else:
            pixel_location_table = np.array([[x.get("easting"), x.get("northing")] for x in rects_match_id])
            linkage_matrix = linkage(pixel_location_table, method='single', metric='chebyshev')

            ctree = cut_tree(linkage_matrix, height=[group_criteria])
            cluster = np.array([x[0] for x in ctree])

        for i in range(len(rects_match_id)):
            rects_match_id[i].update({"defectId": "DEF{:05d}".format(cluster[i] + defect_num)})
        defect_num += max(cluster) + 1

        defect_id_set = set([x.get("defectId") for x in rects_match_id])
        for defect_id in defect_id_set:
            defect = {"defectId": defect_id, "panelGroupId": group_id, "category": DefectCategory.UNCONFIRMED}
            rect_match_defect = [x for x in rects_match_id if x.get("defectId") == defect_id]

            easting = float(np.mean([x.get("easting") for x in rect_match_defect]))
            northing = float(np.mean([x.get("northing") for x in rect_match_defect]))
            severity = float(np.mean([x.get("severity") for x in rects_match_id]))
            utm_zone = rects_match_id[0].get("utm_zone")
            lat, lng = utm.to_latlon(easting, northing, utm_zone, northern=True)
            defect.update({"lat": lat, "lng": lng, "utmEasting": easting, "utmNorthing": northing,
                           "utmZone": utm_zone, "severity": severity})
            defect.update({"rects": [x for x in rect_match_defect]})

            defects.append(defect)

    with open(join(folder_path, "defects.json"), "w") as f:
        json.dump(defects, f)

    return defects


def batch_process_locate(folder_path: str, gsd: float, group_criteria: float) -> List[dict]:
    """
    this function will read all the labeled defects from /labeled/rect.json and output the gps coordinates of the
    defects to a json file folder_path/defects.json. This function also tries to unify those defects that very close
    to each other since they might be the same defect.
    :param folder_path: this is the folder where the raw images rest
    :param gsd: ground sampling distance in meters
    :param group_criteria: the distance criteria of grouping the defects, in meters
    :return: list of defects profile
    """

    with open(join(folder_path, "exif.json"), "r") as f:
        exif = json.load(f)

    with open(join(folder_path, "rect.json"), "r") as f:
        rect_info = json.load(f)

    defects = list()

    for d in rect_info:
        base_name = d.get("image")
        # image_latitude = exif.get(base_name).get("GPSLatitude")
        # image_longitude = exif.get(base_name).get("GPSLongitude")
        cursor = next((x for x in exif if x.get("image") == base_name))
        image_latitude = cursor.get("GPSLatitude")
        image_longitude = cursor.get("GPSLongitude")
        image_height = d.get("height")
        image_width = d.get("width")

        geo_mapper = UTMGeoMapper(gsd=gsd, origin_gps=(image_latitude, image_longitude),
                                  origin_pixel=(image_height / 2 - 0.5, image_width / 2 - 0.5))

        for rect in d.get("rects"):
            x = rect.get("x")
            y = rect.get("y")
            utm_easting, utm_northing, utm_zone = geo_mapper.pixel2utm(row=y, col=x)

            defects.append({"image": base_name, "utm_easting": utm_easting, "utm_northing": utm_northing,
                            "utm_zone": utm_zone, "rect": rect})

            # for image_name, value in rect_info.items():
            #     base_name = os.path.splitext(basename(image_name))[0]
            #     image_latitude = exif.get(base_name).get("GPSLatitude")
            #     image_longitude = exif.get(base_name).get("GPSLongitude")
            #
            #     for rect in value["rects"]:
            #         x_small_image = rect.get("x")
            #         y_small_image = rect.get("y")
            #
            #         x_shift_small_image = x_small_image - (value["width"] - 1) / 2
            #         y_shift_small_image = y_small_image - (value["height"] - 1) / 2
            #
            #         x_shift_large_image = x_shift_small_image / pixel_ratio
            #         y_shift_large_image = y_shift_small_image / pixel_ratio
            #
            #         y_center_large_image, x_center_large_image = geo_mapper.gps2pixel(image_latitude, image_longitude)
            #
            #         x_large_image = x_center_large_image + x_shift_large_image
            #         y_large_image = y_center_large_image + y_shift_large_image
            #
            #         defects.append({"image": base_name, "x_large_image": x_large_image, "y_large_image": y_large_image,
            #                         "rect": rect})

        #     grouping the defects according to pixel distance on the stitched image
    pixel_location_table = np.array([[x.get("utm_easting"), x.get("utm_northing")] for x in defects])
    linkage_matrix = linkage(pixel_location_table, method='single', metric='chebyshev')

    ctree = cut_tree(linkage_matrix, height=[group_criteria])
    cluster = np.array([x[0] for x in ctree])
    cluster_centroids = list()
    for i in range(max(cluster) + 1):
        x_center = np.mean(pixel_location_table[cluster == i, 0])
        y_center = np.mean(pixel_location_table[cluster == i, 1])
        cluster_centroids.append([x_center, y_center])

    for i in range(len(defects)):
        defect = defects[i]
        defect_id_num = cluster[i]
        defect.update({"defect_id_num": defect_id_num})

    clustered_defects = list()
    for i in set(cluster):
        d = dict()
        d.update({
            "defectId": "DEF{:05d}".format(i),
            "utmEasting": cluster_centroids[i][0],
            "utmNorthing": cluster_centroids[i][1],
            "category": DefectCategory.UNCONFIRMED,
            "rects": list()
        })

        involved_defects = [x for x in defects if x.get("defect_id_num") == i]
        d.update({"utmZone": involved_defects[0].get("utm_zone")})

        for involved_defect in involved_defects:
            temp_d = dict()
            temp_d.update(involved_defect.get("rect"))
            temp_d.update({"image": involved_defect.get("image")})
            d.get("rects").append(temp_d)
        lat, lng = utm.to_latlon(d.get("utmEasting"), d.get("utmNorthing"), d.get("utmZone"), northern=True)
        d.update({"lat": lat, "lng": lng})

        clustered_defects.append(d)

    with open(join(folder_path, "defects.json"), "w") as f:
        json.dump(clustered_defects, f)

    return clustered_defects


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    folder_path = r"C:\Users\h232559\Documents\projects\uav\pic\linuo\2017-09-19\ir"
    # batch_process_exif(folder_path)
    batch_process_aggregate(folder_path, group_criteria=2.0)
