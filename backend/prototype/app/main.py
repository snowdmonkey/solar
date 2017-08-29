from flask import Flask, request, send_file
from flask_cors import CORS
from os.path import join
import json
import cv2
import io

app = Flask(__name__)
CORS(app)

defects_summary = None
exif = None


def get_defects_summary() -> dict:
    global defects_summary
    if defects_summary is None:
        with open("C:/Users/h232559/Documents/projects/uav/pic/2017-06-21-funingyilin-DJI/6-21-FLIR/defects.json") as f:
            defects_summary = json.load(f)
    return defects_summary


def get_exif() -> dict:
    global exif
    if exif is None:
        with open(r"C:\Users\h232559\Documents\projects\uav\pic\2017-06-21-funingyilin-DJI\6-21-FLIR\exif.json") as f:
            exif = json.load(f)
    return exif


def get_rotated_folder() -> str:
    """
    :return: the folder where the rotated images are saved
    """
    folder_path = r"C:\Users\h232559\Documents\projects\uav\pic\2017-06-21-funingyilin-DJI\6-21-FLIR\rotated"
    return folder_path


@app.route("/defects")
def get_defects() -> str:
    defects = list()
    for defect_id, defect_info in get_defects_summary().items():
        defect = {"defectId": defect_id,
                  "latitude": defect_info.get("latitude"),
                  "longitude": defect_info.get("longitude")}
        defects.append(defect)
    return json.dumps(defects)


@app.route("/images/defect/<string:defect_id>")
def get_images(defect_id: str) -> str:
    """
    return a json string which contains the names of the images relating to a defect
    :param defect_id: string starts with "defect" and followed by an int, e.g., defect146
    :return: json string
    """
    defect_info = get_defects_summary().get(defect_id)
    image_names = defect_info.get("images")
    results = list()
    for image_name in image_names:
        latitude = get_exif().get(image_name).get("GPSLatitude")
        longitude = get_exif().get(image_name).get("GPSLongitude")
        results.append({"imageName": image_name, "latitude": latitude, "longitude": longitude})
    return json.dumps(results)


@app.route("/image/labeled")
def get_labeled_image():
    """
    generate and return an image with image name and defect id, the corresponding defects should be labeled on the image
    :return:
    """
    image_name = request.args.get("image")
    defect_id = request.args.get("defect")
    rects = get_defects_summary().get(defect_id).get("images").get(image_name)
    img = cv2.imread(join(get_rotated_folder(), image_name), cv2.IMREAD_COLOR)
    for rect in rects:
        x, y, w, h = rect.get("x"), rect.get("y"), rect.get("w"), rect.get("h")
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    img_bytes = cv2.imencode(".png", img)[1]
    return send_file(io.BytesIO(img_bytes), attachment_filename="labeled.png", mimetype="image/png")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)