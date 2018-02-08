from pathlib import Path
import numpy as np
import cv2
import subprocess
import json


def overlay_image(raw: np.ndarray, anno: np.ndarray, alpha: float) -> np.ndarray:
    out = raw.copy()
    out = cv2.addWeighted(out, 1-alpha, anno, alpha, 0)
    return out


image_folder = Path(r"C:\Users\h232559\Documents\projects\uav\label\2017-06-20")

label_folder = Path(r"C:\Users\h232559\Documents\projects\uav\label\2017-06-20\Pixel Labeled Images\Panel_HotSpot_Rest")

anno_folder = image_folder / "annotation"
overlay_folder = image_folder / "overlay"

if not anno_folder.exists():
    anno_folder.mkdir()

if not overlay_folder.exists():
    overlay_folder.mkdir()

for img_path in image_folder.glob("*.jpg"):
    label_path = label_folder / img_path.name.replace(".jpg", ".png")
    anno_path = anno_folder / img_path.name.replace(".jpg", ".png")
    overlay_path = overlay_folder / img_path.name
    if not label_path.exists():
        continue

    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

    # cmd = ["exiftool", "-FlightYawDegree", "-j", str(img_path)]
    # proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    # d = json.loads(proc.stdout.decode("utf-8"))[0]
    # if d.get("FlightYawDegree") > 0:
    #     rows, cols = label.shape
    #     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
    #     label = cv2.warpAffine(label, M, (cols, rows))

    panel_group_mask = np.zeros_like(label)

    temp_mask = np.zeros_like(label)
    temp_mask[(label == 1) | (label == 2)] = 255

    _, cnts, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        continue

    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 100]

    defect_mask = np.zeros_like(label)
    defect_mask[label == 2] = 255

    raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    for cnt in cnts:
        panel_mask = np.zeros_like(label)
        cv2.drawContours(panel_mask, [cnt], -1, 255, -1)

        panel_group_mask[panel_mask == 255] = 255

        panel_temp = raw[panel_mask == 255]
        limits = np.percentile(panel_temp, (10, 90))
        trimmed_temp = panel_temp[(limits[0] < panel_temp) & (panel_temp < limits[1])]
        # trimmed_data = [x for x in data if limits[0] < x < limits[1]]
        mu = np.mean(trimmed_temp)
        sd = np.std(trimmed_temp)
        threshold = mu + 2.0 * sd

        th_mask = np.zeros_like(label)
        th_mask[raw > threshold] = 255

        defect_mask[(th_mask == 0) & (panel_mask == 255)] = 0

    out = np.zeros(shape=label.shape+(3,), dtype=np.uint8)
    out[panel_group_mask == 255] = (0, 255, 0)
    out[defect_mask == 255] = (0, 0, 255)
    cv2.imwrite(str(anno_path), out)

    raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    overlay = overlay_image(raw, out, alpha=0.3)
    cv2.imwrite(str(overlay_path), overlay)
