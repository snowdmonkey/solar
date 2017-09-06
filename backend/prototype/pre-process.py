from os import listdir
from os.path import isfile, join
from extract_rect import rotate_and_scale, PanelCropper
from detect_hotspot import HotSpotDetector
from geomapping.geo_mapper import GeoMapper, AnchorGeoMapper
from scipy.cluster.hierarchy import linkage, cut_tree
import os
import subprocess
import exifread
import json
import cv2
import numpy as np


def convert_gps(gps_info):
    info = str(gps_info)[1:-1]
    a, b, c = info.split(',')
    ret = float(a) + float(b) / 60
    d, e = c.split('/')
    ret += float(d) / float(e) / 60 / 60
    return ret


def process_exif(image, is_visual, output):
    imgf = open(image, 'rb')
    tags = exifread.process_file(imgf)
    rec = '"%s", %s, %s, %s, %s\n' \
          % (image, tags.get('EXIF DateTimeOriginal'), convert_gps(tags.get('GPS GPSLongitude')),
             convert_gps(tags.get('GPS GPSLatitude')), is_visual)
    output.write(rec)


def load_images(image_folder, is_visual, output):
    """Loads the images to database
    :param image_folder: the string representing the folder holding the image files
    """
    images = [join(image_folder, f) for f in listdir(image_folder) if
              isfile(join(image_folder, f)) and f.lower().endswith('.jpg')]
    for i in images:
        process_exif(i, is_visual, output)


def batch_process_exif(folder_path, outfile_path=None):
    """
    use exiftool to extract exif information, include the camera's gps, relative altitude, gesture
    :param folder_path: the path of folder that contains the IR images
    :param outfile_path: the path of json file to store the exif information, default to be under folder_path
    :return: a dictionary with key as the image name and value as the exif information
    """
    if outfile_path is None:
        outfile_path = join(folder_path, "exif.json")
    file_names = [x for x in listdir(folder_path) if x.endswith(".jpg")]
    exif = dict()
    for file_name in file_names:
        command = 'exiftool -j -c "%+.10f" '
        file_path = join(folder_path, file_name)
        proc = subprocess.run(command + file_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        out = proc.stdout
        r_json = json.loads(out.decode("utf-8"))[0]
        exif[file_name] = dict()
        exif[file_name]["DateTimeOriginal"] = r_json.get("DateTimeOriginal")
        exif[file_name]["GPSLatitude"] = float(r_json.get("GPSLatitude"))
        exif[file_name]["GPSLongitude"] = float(r_json.get("GPSLongitude"))
        exif[file_name]["RelativeAltitude"] = r_json.get("RelativeAltitude")
        exif[file_name]["GimbalRollDegree"] = r_json.get("GimbalRollDegree")
        exif[file_name]["GimbalYawDegree"] = r_json.get("GimbalYawDegree")
        exif[file_name]["GimbalPitchDegree"] = r_json.get("GimbalPitchDegree")
    with open(outfile_path, "w") as outfile:
        json.dump(exif, outfile)
    return exif


def batch_process_rotation(folder_path, exif_path=None):
    """
    the function will create a sub folder under folder_path which contains the rotated images. And the images under
    folder path will be rotated so they will all head north. And the degrees for rotate each image should be provided
    in a json file at exif_path with key "GimbalYawDegree"
    :param folder_path: path of the folder contains raw images
    :param exif_path: path of the json file contain exif information, defaults to be under folder_path
    :return: nothing
    """
    rotate_folder_path = join(folder_path, "rotated")
    if not os.path.exists(rotate_folder_path):
        os.mkdir(rotate_folder_path)
    if exif_path is None:
        exif_path = join(folder_path, "exif.json")
    with open(exif_path) as file:
        exif = json.load(file)
    file_names = [x for x in listdir(folder_path) if x.endswith(".jpg")]
    for file_name in file_names:
        image_path = join(folder_path, file_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        degree = exif.get(file_name).get("GimbalYawDegree")
        if degree is not None:
            if abs(degree) < 10:
                rotated_img = img
            elif abs(degree - 90.0) < 10:
                rotated_img = rotate_and_scale(img, -90.0)
            elif abs(degree + 90.0) < 10:
                rotated_img = rotate_and_scale(img, 90.0)
            elif abs(degree - 180.0) < 10:
                rotated_img = rotate_and_scale(img, 180.0)
            elif abs(degree + 180.0) < 10:
                rotated_img = rotate_and_scale(img, 180.0)
            else:
                continue
            rotated_img_path = join(rotate_folder_path, file_name)
            cv2.imwrite(rotated_img_path, rotated_img)


def batch_process_label(folder_path):
    """
    process the images under the rotated sub-directory of folder_path, label the defects with red rectangle
    :param folder_path: folder for the raw images
    :return: nothing
    """
    rotate_folder_path = join(folder_path, "rotated")
    label_folder_path = join(folder_path, "labeled")
    rect_dict = dict()
    if not os.path.exists(label_folder_path):
        os.mkdir(label_folder_path)
    file_names = [x for x in listdir(rotate_folder_path) if x.endswith(".jpg")]
    for file_name in file_names:
        file_path = join(rotate_folder_path, file_name)
        panel_cropper = PanelCropper(file_path)
        sub_imgs = panel_cropper.get_sub_imgs(rotate_n_crop=False, min_area=5000,
                                              verify_rectangle=10, n_vertices_threshold=8)
        if len(sub_imgs) == 0:
            continue

        else:
            points = list()
            for img in sub_imgs:
                hot_spot_detector = HotSpotDetector(img, 3.0)
                hot_spot = hot_spot_detector.get_hot_spot()
                points.extend(hot_spot.points)

            mask = np.zeros_like(panel_cropper.raw_img)
            for point in points:
                mask[tuple(point)] = 255

            _, contours, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cv2.convexHull(x) for x in contours if cv2.contourArea(x) > 2]

            if len(contours) == 0:
                continue
            raw_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            rectangles = list()
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h < 200*w:
                    rectangles.append((x, y, w, h))
            if len(rectangles) > 0:
                rect_dict[file_name] = list()
                for rectangle in rectangles:
                    x, y, w, h = rectangle
                    cv2.rectangle(raw_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    rect_dict[file_name].append({"x": x, "y": y, "w": w, "h": h})

                labeled_img_path = join(label_folder_path, file_name)
                cv2.imwrite(labeled_img_path, raw_image)
    outfile_path = join(label_folder_path, "rect.json")
    with open(outfile_path, "w") as file:
        json.dump(rect_dict, file)


def batch_process_locate(folder_path: str, geo_mapper: GeoMapper, pixel_ratio: float) -> None:
    """
    this function will read all the labeled defects from /labeled/rect.json and output the gps coordinates of the
    defects to a json file folder_path/defects.json. This function is also try to unify those defects that very close
    to each other since they might be the same defect.
    :param folder_path: this is the folder where the raw images rest
    :param geo_mapper: the geo_mapper that can map a pixel on the big map to a pair of gps coordinates and vice versa
    :param pixel_ratio: the number of pixels on the small map that is equivalent to one pixel on the big map in term
    of the same amount of the physical distance
    :return: nothing
    """

    with open(join(folder_path, "exif.json"), "r") as f:
        exif = json.load(f)

    with open(join(folder_path, "labeled", "rect.json"), "r") as f:
        rect_info = json.load(f)

    defects = list()
    for image_name, rects in rect_info.items():
        image_latitude = exif.get(image_name).get("GPSLatitude")
        image_longitude = exif.get(image_name).get("GPSLongitude")

        for rect in rects:
            x_small_image = rect.get("x")
            y_small_image = rect.get("y")

            x_shift_small_image = x_small_image - 128
            y_shift_small_image = y_small_image - 168

            x_shift_large_image = x_shift_small_image / pixel_ratio
            y_shift_large_image = y_shift_small_image / pixel_ratio

            y_center_large_image, x_center_large_image = geo_mapper.gps2pixel(image_latitude, image_longitude)

            x_large_image = x_center_large_image + x_shift_large_image
            y_large_image = y_center_large_image + y_shift_large_image

            defects.append({"image": image_name, "x_large_image": x_large_image, "y_large_image": y_large_image,
                            "rect": rect})

#     grouping the defects according to pixel distance on the stitched image
    pixel_location_table = np.array([[x.get("x_large_image"), x.get("y_large_image")] for x in defects])
    linkage_matrix = linkage(pixel_location_table, method='single', metric='chebyshev')
    ctree = cut_tree(linkage_matrix, height=[5])
    cluster = np.array([x[0] for x in ctree])
    cluster_centroids = list()
    for i in range(max(cluster)+1):
        x_center = np.mean(pixel_location_table[cluster == i, 0])
        y_center = np.mean(pixel_location_table[cluster == i, 1])
        cluster_centroids.append([x_center, y_center])

    clustered_defects = dict()

    for i in range(len(defects)):
        defect = defects[i]
        defect_id_num = cluster[i]
        defect_id = "defect" + str(defect_id_num)
        if clustered_defects.get(defect_id) is None:
            clustered_defects[defect_id] = dict()
            clustered_defects[defect_id]["x"] = round(cluster_centroids[defect_id_num][0])
            clustered_defects[defect_id]["y"] = round(cluster_centroids[defect_id_num][1])
            clustered_defects[defect_id]["latitude"], clustered_defects[defect_id]["longitude"] = \
                geo_mapper.pixel2gps(cluster_centroids[defect_id_num][1], cluster_centroids[defect_id_num][0])
            clustered_defects[defect_id]["images"] = dict()
            clustered_defects[defect_id]["images"][defect.get("image")] = [defect.get("rect")]
        elif clustered_defects[defect_id]["images"].get(defect.get("image")) is None:
            clustered_defects[defect_id]["images"][defect.get("image")] = [defect.get("rect")]
        else:
            clustered_defects[defect_id]["images"][defect.get("image")].append(defect.get("rect"))

    with open(join(folder_path, "defects.json"), "w") as f:
        json.dump(clustered_defects, f)


if __name__ == '__main__':
    # of = open('C:\\SolarPanel\\2017-06-20\\exif.csv', 'w')
    # load_images('C:\\SolarPanel\\2017-06-20\\6-20-DJI', True, of)
    # load_images('C:\\SolarPanel\\2017-06-20\\6-20-FLIR', False, of)
    # of.close()
    # of = open('C:\\SolarPanel\\2017-06-21\\exif.csv', 'w')
    # load_images('C:\\SolarPanel\\2017-06-21\\6-21-FLIR', False, of)
    # of.close()
    # of = open('C:\\SolarPanel\\2017-07-04\\exif.csv', 'w')
    # load_images('C:\\SolarPanel\\2017-07-04\\7-04-1', True, of)
    # load_images('C:\\SolarPanel\\2017-07-04\\7-04-2', True, of)
    # of.close()
    folder_path = r"C:\Users\h232559\Documents\projects\uav\pic\2017-06-21-funingyilin-DJI\6-21-FLIR"

    pixel_anchors = [[639, 639], [639, 1328], [639, 2016],
                     [1358, 639], [1358, 1328], [1358, 2016],
                     [2076, 639], [2076, 1328], [2076, 2016]]
    gps_anchors = [[33.59034075, 119.63160525], [33.59034075, 119.6334535], [33.59034075, 119.63530175],
                   [33.58873250, 119.63160525], [33.58873250, 119.6334535], [33.58873250, 119.63530175],
                   [33.58712425, 119.63160525], [33.58712425, 119.6334535], [33.58712425, 119.63530175]]

    geo_mapper = AnchorGeoMapper(pixel_anchors=pixel_anchors, gps_anchors=gps_anchors)

    batch_process_locate(folder_path, geo_mapper, 5.3793)
