from flask import Flask
from flask_cors import CORS
import json

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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)