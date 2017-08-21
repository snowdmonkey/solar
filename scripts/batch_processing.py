from os import listdir
from os.path import join, basename, splitext
import exifread
import cv2
import numpy as np
from matplotlib import pyplot as plt
from extract_rect import PanelCropper
from detect_hotspot import HotSpotDetector

folder_path = r"C:\Users\h232559\Documents\projects\uav\pic\8-15\zone1-ir-15m"
file_names = [join(folder_path, x) for x in listdir(folder_path) if x.endswith(".jpg")]
# [print(x) for x in file_names]
for file_name in file_names:

    print(file_name)
    with open(file_name, "rb") as file:
                tags = exifread.process_file(file)
    altitude_ratio = tags.get("GPS GPSAltitude").values[0]
    altitude = altitude_ratio.num / altitude_ratio.den

    if altitude < 700.0:
        panel_cropper = PanelCropper(file_name)
        sub_imgs = panel_cropper.get_sub_imgs(rotate_n_crop=False, min_area=5000,
                                              verify_rectangle=10, n_vertices_threshold=8)

        if len(sub_imgs) == 0:
            contours = list()
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

        plt.figure()
        raw_image = cv2.imread(file_name)
        plt.subplot(1, 2, 1)
        plt.imshow(raw_image)
        plt.xticks([]), plt.yticks([])
        plt.subplot(1, 2, 2)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(raw_image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        plt.imshow(raw_image)
        plt.xticks([]), plt.yticks([])
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.savefig("../processed_pic/processed_pic_20170815/" + basename(file_name),
                    dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close()

