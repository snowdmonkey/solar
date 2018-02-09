import json
import cv2
import numpy as np
from pathlib import Path

folders = [Path(r"C:\Users\h232559\Documents\projects\uav\label\2017-06-20"),
           Path(r"C:\Users\h232559\Documents\projects\uav\label\2017-06-21"),
           Path(r"C:\Users\h232559\Documents\projects\uav\label\2017-08-15"),
           Path(r"C:\Users\h232559\Documents\projects\uav\label\2017-09-19")]

results = list()

for folder in folders:
    anno_folder = folder / "annotation"
    bbox_folder = folder / "bbox"
    if not bbox_folder.exists():
        bbox_folder.mkdir()
    json_path = folder / "bbox.json"

    for file_path in folder.glob("*.jpg"):
        anno_path = (anno_folder / file_path.name).with_suffix(".png")

        if not anno_path.exists():
            continue

        anno = cv2.imread(str(anno_path), cv2.IMREAD_COLOR)
        img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)

        height, width, _ = anno.shape

        panel_mask = np.zeros(shape=(height, width), dtype=np.uint8)
        panel_mask[(anno[:, :, 0] == 0) & (anno[:, :, 1] == 255) & (anno[:, :, 2] == 0)] = 255

        _, cnts, _ = cv2.findContours(panel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

            results.append({"filename": file_path.name,
                            "width": width,
                            "height": height,
                            "class": "panel",
                            "xmin": x,
                            "xmax": x+w,
                            "ymin": y,
                            "ymax": y+h})

        defect_mask = np.zeros(shape=(height, width), dtype=np.uint8)
        defect_mask[(anno[:, :, 0] == 0) & (anno[:, :, 1] == 0) & (anno[:, :, 2] == 255)] = 255
        kernel = np.ones((3, 3), np.uint8)
        defect_mask = cv2.dilate(defect_mask, kernel, iterations=1)

        _, cnts, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

            results.append({"filename": file_path.name,
                            "width": width,
                            "height": height,
                            "class": "defect",
                            "xmin": x,
                            "xmax": x + w,
                            "ymin": y,
                            "ymax": y + h})

        cv2.imwrite(str(bbox_folder / file_path.name.replace(".jpg", ".png")), img)

    with json_path.open("w") as f:
        json.dump(results, f)




