from flask import Flask, request, send_file, abort, jsonify
from flask_cors import CORS
from os.path import join
from image_process import ImageProcessPipeline
from pymongo import MongoClient
from typing import Union
import cv2
import io
import logging
import os

app = Flask(__name__)
CORS(app)

# defects_summary = None
# exif = None
image_root_path = None
mongo_client = None


def get_mongo_client() -> MongoClient:
    global mongo_client
    if mongo_client is None:
        mongo_host = os.getenv("MONGO_HOST", "mongo")
        mongo_client = MongoClient(host=mongo_host, port=27017)
    return mongo_client


def get_image_root() -> str:
    global image_root_path
    if image_root_path is None:
        image_root_path = os.getenv("IMG_ROOT")
    return image_root_path


def get_defects_summary(station: str, date: str) -> Union[None, dict]:

    value = get_mongo_client().solar.defect.find_one({"date": date, "station": station}, {"value": 1})
    if value is None:
        defects_summary = None
    else:
        defects_summary = value.get("value")
    return defects_summary


def get_exif(station: str, date: str, image: str) -> Union[dict, None]:

    exif = get_mongo_client().solar.exif.find_one({"station": station, "date": date, "image": image}, {"_id": 0})
    # if value is None:
    #     exif = None
    # else:
    #     exif = value.get("value")
    return exif


def get_rotated_folder(station: str, date: str) -> str:
    """
    :return: the folder where the rotated images are saved
    """
    folder_path = join(get_image_root(), station, date, "ir/rotated")
    return folder_path


def get_visual_folder(station: str, date: str) -> str:
    """
    :return: the folder where the raw visual images are saved
    """
    folder_path = join(get_image_root(), station, date, "visual/rotated")
    return folder_path


@app.route("/station/<str:station>/date", methos=["GET"])
def get_reports_date_by_station(station: str):
    """
    return the dates of available reports for a station
    """
    posts = get_mongo_client().solar.defect.find({"station": station}, {"date": True})
    result = [x.get("date") for x in posts]
    return jsonify(result)


@app.route("/station/<str:station>/date/<str:date>/defect", methods=["GET"])
def get_defect_by_date_and_station(station: str, date: str):
    """
    return a defect list by station and date
    """
    defect_summary = get_defects_summary(station, date)
    if defect_summary is None:
        abort(404)
    else:
        defects = list()
        for defect_id, defect_info in defect_summary.items():
            defect = {
                "defectId": defect_id,
                "latitude": defect_info.get("latitude"),
                "longitude": defect_info.get("longitude"),
                "category": defect_info.get("category"),
                "groupId": defect_info.get("group")
            }
            defects.append(defect)
        return jsonify(defects)


@app.route("/station/<str:station>/date/<str:date>/defect", methods=["PUT"])
def analyze_by_date_and_station(station: str, date: str):
    folder_path = join(get_image_root(), station, date)
    pipeline = ImageProcessPipeline(image_folder=folder_path, station=station, date=date)
    pipeline.run()
    return "OK"


@app.route("/station/<str:station>/date/<str:date>/defect/<str:defect_id>", methods=["PUT"])
def set_defect_by_id(station: str, date: str, defect_id: str):
    """
    set a defect's gps coordinates and category
    """
    post_body = request.get_json()
    lat = post_body.get("lat")
    lng = post_body.get("lng")
    cat = post_body.get("category")
    defect_collection = get_mongo_client().solar.defect

    if defect_collection.find_one({"station": station, "date": date}).get("value").get(defect_id) is None:
        abort(404)

    if lat is not None:
        defect_collection.update_one({"station": station, "date": date},
                                     {"$set": {"value.{}.latitude".format(defect_id): lat}})
    if lng is not None:
        defect_collection.update_one({"station": station, "date": date},
                                     {"$set": {"value.{}.longitude".format(defect_id): lng}})
    if cat is not None:
        defect_collection.update_one({"station": station, "date": date},
                                     {"$set": {"value.{}.category".format(defect_id): cat}})
    return "OK"


@app.route("/station/<str:station>/date/<str:date>/defect/<str:defect_id>/image", methods=["GET"])
def get_images_by_defect(station: str, date: str, defect_id: str):
    """
    return a json string which contains the names of the images relating to a defect
    :return: json string
    """
    defect_info = get_defects_summary(station, date).get(defect_id)
    image_names = defect_info.get("images")
    results = list()
    # exif = get_exif(station, date)
    # if exif is None:
    #     abort(404)
    for image_name in image_names:
        # latitude = exif.get(image_name).get("GPSLatitude")
        # longitude = exif.get(image_name).get("GPSLongitude")
        exif = get_exif(station=station, date=date, image=image_name)
        lat = exif.get("GPSLatitude")
        lng = exif.get("GPSLongitude")
        results.append({"imageName": image_name, "latitude": lat, "longitude": lng})
    return jsonify(results)


@app.route("/station/<str:station>/date/<str:date>/image/ir/<str:image>", methods=["GET"])
def get_labeled_image(station: str, date: str, image: str):
    """
    generate and return an image with image name and defect id, the corresponding defects should be labeled on the image
    label a defect if a defect_id is provided
    :return:
    """
    base_image_name = image
    image_name = base_image_name + ".jpg"
    defect_id = request.args.get("defect")
    color_map = request.args.get("colorMap")

    img = cv2.imread(join(get_rotated_folder(station, date), image_name), cv2.IMREAD_COLOR)

    mapping = {
        "autumn": cv2.COLORMAP_AUTUMN,
        "bone": cv2.COLORMAP_BONE,
        "jet": cv2.COLORMAP_JET,
        "winter": cv2.COLORMAP_WINTER,
        "rainbow": cv2.COLORMAP_RAINBOW,
        "ocean": cv2.COLORMAP_OCEAN,
        "summer": cv2.COLORMAP_SUMMER,
        "spring": cv2.COLORMAP_SPRING,
        "cool": cv2.COLORMAP_COOL,
        "hsv": cv2.COLORMAP_HSV,
        "pink": cv2.COLORMAP_PINK,
        "hot": cv2.COLORMAP_HOT}

    rect_color = (0, 0, 255)

    if color_map is not None:
        rect_color = (255, 255, 255)
        color_id = mapping.get(color_map)
        if color_id is None:
            abort(400, "illegal color map")
        else:
            img = cv2.applyColorMap(img, color_id)

    if defect_id is not None:
        rects = get_defects_summary(station=station, date=date).get(defect_id).get("images").get(base_image_name)
        for rect in rects:
            x, y, w, h = rect.get("x"), rect.get("y"), rect.get("w"), rect.get("h")
            cv2.rectangle(img, (x, y), (x + w, y + h), rect_color, 1)

    img_bytes = cv2.imencode(".png", img)[1]
    return send_file(io.BytesIO(img_bytes), attachment_filename="labeled.png", mimetype="image/png")


@app.route("/station/<str:station>/date/<str:date>/image/visual/<str:image>", methods=["GET"])
def get_visual_image(station: str, date: str, image: str):
    """
    :return: raw visual image specified by image name
    """
    base_image_name = image
    image_name = base_image_name + ".jpg"
    img = cv2.imread(join(get_visual_folder(station=station, date=date), image_name), cv2.IMREAD_COLOR)
    img_bytes = cv2.imencode(".png", img)[1]
    return send_file(io.BytesIO(img_bytes), attachment_filename="visual.png", mimetype="image/png")


@app.route("/panel_group", methods=["GET"])
def get_panel_groups():
    """
    get all the panel groups positions
    :return:
    """
    db = get_mongo_client()
    cursor = db.solar.panelGroup.find(filter={}, projection={"_id": False})
    result = list()
    for post in cursor:
        result.append(post)
    return jsonify(result)


@app.route("/panel_group", methods=["POST"])
def add_panel_group():
    """
    add a new panel group
    :return:
    """
    db = get_mongo_client()
    post = request.get_json()
    post["id"] = str(post["id"])
    if db.solar.panelGroup.find_one({"id": post.get("id")}) is not None:
        abort(400, "naming conflict")
    db.solar.panelGroup.update_one({"id": post.get("id")}, {"$set": {"corners": post.get("corners")}}, upsert=True)
    return "OK"


@app.route("/panel_group/<string:group_id>", methods=["GET"])
def get_panel_group(group_id: str):
    """
    get the details of a panel group
    :param group_id:
    :return:
    """
    db = get_mongo_client()
    result = db.solar.panelGroup.find_one(filter={"id": group_id}, projection={"_id": False})
    return jsonify(result)


@app.route("/panel_group/<string:group_id>", methods=["PUT"])
def set_panel_group(group_id: str):
    """
    set the name and/or the corners of a panel group
    :param group_id:
    :return:
    """
    db = get_mongo_client()
    post = request.get_json()

    if db.solar.panelGroup.find_one({"id": group_id}) is None:
        abort(404, "id not found")

    if post.get("id") is not None:
        if post.get("id") != group_id:
            current = db.solar.panelGroup.find_one({"id": post.get("id")})
            if current is not None:
                abort(400, "naming conflict")
            else:
                db.solar.panelGroup.update_one({"id": group_id}, {"$set": {"id": post.get("id")}})
        if post.get("corners") is not None:
            db.solar.panelGroup.update_one({"id": post.get("id")}, {"$set": {"corners": post.get("corners")}})
    elif post.get("corners") is not None:
        db.solar.panelGroup.update_one({"id": group_id}, {"$set": {"corners", post.get("corners")}})
    return "OK"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("log"), logging.StreamHandler()])

    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
