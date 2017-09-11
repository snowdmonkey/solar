from flask import Flask, request, send_file
from flask_cors import CORS
from os.path import join
from image_process import ImageProcessPipeline
import os
import json
import cv2
import io
import logging

app = Flask(__name__)
CORS(app)

# defects_summary = None
# exif = None
image_root_path = None


def get_image_root() -> str:
    global image_root_path
    if image_root_path is None:
        image_root_path = os.getenv("IMG_ROOT")
    return image_root_path


def get_defects_summary(date: str) -> dict:
    with open(join(get_image_root(), date, "ir/defects.json")) as f:
        defects_summary = json.load(f)
    return defects_summary


def get_exif(date: str) -> dict:
    with open(join(get_image_root(), date, "ir/exif.json")) as f:
        exif = json.load(f)
    return exif


def get_rotated_folder(date: str) -> str:
    """
    :return: the folder where the rotated images are saved
    """
    folder_path = join(get_image_root(), date, "ir/rotated")
    return folder_path


def get_visual_folder(date: str) -> str:
    """
    :return: the folder where the raw visual images are saved
    """
    folder_path = join(get_image_root(), date, "visual/rotated")
    return folder_path


@app.route("/defects", methods=["POST", "GET"])
def get_defects() -> str:
    if request.method == "GET":
        defects = list()
        date = request.args.get("date")
        for defect_id, defect_info in get_defects_summary(date).items():
            defect = {"defectId": defect_id,
                      "latitude": defect_info.get("latitude"),
                      "longitude": defect_info.get("longitude")}
            defects.append(defect)
        return json.dumps(defects)
    elif request.method == "POST":
        date = request.form.get("date")
        folder_path = join(get_image_root(), date)
        pipeline = ImageProcessPipeline(image_folder=folder_path, date=date)
        pipeline.run()
        return "success"


@app.route("/images/defect")
def get_images() -> str:
    """
    return a json string which contains the names of the images relating to a defect
    :return: json string
    """
    date = request.args.get("date")
    defect_id = request.args.get("defectId")
    defect_info = get_defects_summary(date).get(defect_id)
    image_names = defect_info.get("images")
    results = list()
    for image_name in image_names:
        latitude = get_exif(date).get(image_name).get("GPSLatitude")
        longitude = get_exif(date).get(image_name).get("GPSLongitude")
        results.append({"imageName": image_name, "latitude": latitude, "longitude": longitude})
    return json.dumps(results)


@app.route("/image/labeled")
def get_labeled_image():
    """
    generate and return an image with image name and defect id, the corresponding defects should be labeled on the image
    :return:
    """
    base_image_name = request.args.get("image")
    image_name = base_image_name + ".jpg"
    defect_id = request.args.get("defect")
    date = request.args.get("date")
    rects = get_defects_summary(date).get(defect_id).get("images").get(base_image_name)
    img = cv2.imread(join(get_rotated_folder(date), image_name), cv2.IMREAD_COLOR)
    for rect in rects:
        x, y, w, h = rect.get("x"), rect.get("y"), rect.get("w"), rect.get("h")
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    img_bytes = cv2.imencode(".png", img)[1]
    return send_file(io.BytesIO(img_bytes), attachment_filename="labeled.png", mimetype="image/png")


@app.route("/image/visual")
def get_visual_image():
    """
    :return: raw visual image specified by image name
    """
    base_image_name = request.args.get("image")
    image_name = base_image_name + ".jpg"
    date = request.args.get("date")
    img = cv2.imread(join(get_visual_folder(date), image_name), cv2.IMREAD_COLOR)
    img_bytes = cv2.imencode(".png", img)[1]
    return send_file(io.BytesIO(img_bytes), attachment_filename="visual.png", mimetype="image/png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("log"), logging.StreamHandler()])

    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)