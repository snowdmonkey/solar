from os import listdir
from os.path import isfile, join
from extract_rect import rotate_and_scale
import os
import subprocess
import exifread
import json
import cv2


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
        command = 'exiftool.exe -j -c "%+.10f" '
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
            if abs(degree) < 5:
                rotated_img = img
            elif abs(degree - 90.0) < 5:
                rotated_img = rotate_and_scale(img, -90.0)
            elif abs(degree + 90.0) < 5:
                rotated_img = rotate_and_scale(img, 90.0)
            elif abs(degree - 180.0) < 5:
                rotated_img = rotate_and_scale(img, 180.0)
            elif abs(degree + 180.0) < 5:
                rotated_img = rotate_and_scale(img, 180.0)
            else:
                continue
            rotated_img_path = join(rotate_folder_path, file_name)
            cv2.imwrite(rotated_img_path, rotated_img)



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
    folder_path = r"C:/Users/h232559/Documents/projects/uav/pic/8-15/zone1-ir-15m"
    batch_process_rotation(folder_path)
