from flask import Flask, request, send_file, abort, jsonify
from flask_cors import CORS
from os.path import join
from image_process import ImageProcessPipeline
from pymongo import MongoClient
from typing import Union, List
from temperature import TempTransformer
import numpy as np
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


def get_defects_summary(station: str, date: str) -> Union[None, List[dict]]:

    value = get_mongo_client().solar.defect.find({"date": date, "station": station}, {"_id": False})
    if value is None:
        defects_summary = None
    else:
        defects_summary = value
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


@app.route("/api/v1/station", methods=["GET"])
def get_station_list():
    """
    :return: a list of stations
    """
    posts = get_mongo_client().solar.station.find({}, {"_id": False})
    results = [x for x in posts]
    return jsonify(results)


@app.route("/api/v1/station", methods=["POST"])
def add_station():
    """
    add a station
    """
    collection = get_mongo_client().get_database("solar").get_collection("station")
    post_body = request.get_json()
    station_id = post_body.get("stationId")
    station_name = post_body.get("stationName")
    station_description = post_body.get("description")
    station_gps = post_body.get("gps")
    if station_id is None:
        abort(400, "need to provide station ID")
    elif station_name is None:
        abort(400, "need station name")
    elif station_gps is None:
        abort(400, "need gps")
    else:
        collection.insert_one({
            "stationId": station_id,
            "stationName": station_name,
            "description": station_description,
            "gps": station_gps
        })
        return "OK"


@app.route("/api/v1/station/<string:station>", methods=["GET"])
def get_station_by_id(station):
    """
    return the profile of a station
    """
    result = get_mongo_client().solar.station.find_one({"stationId": station}, {"_id": False})
    if result is None:
        abort(404)
    else:
        return jsonify(result)


@app.route("/api/v1/station/<string:station>/date", methods=["GET"])
def get_reports_by_date_station(station: str):
    """
    return the dates of available reports for a station
    """
    collection = get_mongo_client().get_database("solar").get_collection("defect")
    posts = collection.find({"station": station}, {"date": True}).distinct(key="date")
    return jsonify(posts)


@app.route("/api/v1/station/<string:station>/date/<string:date>/defect", methods=["GET"])
def get_defect_by_date_and_station(station: str, date: str):
    """
    return a defect list by station and date
    """
    defect_summary = get_defects_summary(station, date)
    if defect_summary is None:
        abort(404)
    else:
        defects = list()
        for d in defect_summary:
            defect = {
                "defectId": d.get("defectId"),
                "latitude": d.get("lat"),
                "longitude": d.get("lng"),
                "category": d.get("category"),
                "groupId": d.get("group")
            }
            defects.append(defect)
        return jsonify(defects)


@app.route("/api/v1/station/<string:station>/date/<string:date>/defect", methods=["PUT"])
def analyze_by_date_and_station(station: str, date: str):
    folder_path = join(get_image_root(), station, date)
    pipeline = ImageProcessPipeline(image_folder=folder_path, station=station, date=date)
    pipeline.run()
    return "OK"


@app.route("/api/v1/station/<string:station>/date/<string:date>/defect/<string:defect_id>", methods=["PUT"])
def set_defect_by_id(station: str, date: str, defect_id: str):
    """
    set a defect's gps coordinates and category
    """
    post_body = request.get_json()
    lat = post_body.get("lat")
    lng = post_body.get("lng")
    cat = post_body.get("category")
    defect_collection = get_mongo_client().solar.defect

    if defect_collection.find_one({"station": station, "date": date, "defectId": defect_id}) is None:
        abort(404)

    if lat is not None:
        defect_collection.update_one({"station": station, "date": date, "defectId": defect_id},
                                     {"$set": {"lat": lat}})
    if lng is not None:
        defect_collection.update_one({"station": station, "date": date, "defectId": defect_id},
                                     {"$set": {"lng": lng}})
    if cat is not None:
        defect_collection.update_one({"station": station, "date": date, "defectId": defect_id},
                                     {"$set": {"category".format(defect_id): cat}})
    return "OK"


@app.route("/api/v1/station/<string:station>/date/<string:date>/defect/<string:defect_id>/image", methods=["GET"])
def get_images_by_defect(station: str, date: str, defect_id: str):
    """
    return a json string which contains the names of the images relating to a defect
    :return: json string
    """
    defect_info = get_mongo_client().get_database("solar")\
        .get_collection("defect")\
        .find_one({"station": station, "date": date, "defectId": defect_id})
    image_names = [x.get("image") for x in defect_info.get("rects")]

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


@app.route("/api/v1/station/<string:station>/date/<string:date>/image/ir/<string:image>", methods=["GET"])
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
        # rects = get_defects_summary(station=station, date=date).get(defect_id).get("images").get(base_image_name)
        rects = get_mongo_client().get_database("solar").get_collection("defect")\
            .find_one({"station": station, "date": date, "defectId": defect_id}, {"_id": 0, "rects": 1}).get("rects")
        for rect in rects:
            if rect.get("image") == image:
                x, y, w, h = rect.get("x"), rect.get("y"), rect.get("w"), rect.get("h")
                cv2.rectangle(img, (x, y), (x + w, y + h), rect_color, 1)

    img_bytes = cv2.imencode(".png", img)[1]
    return send_file(io.BytesIO(img_bytes), attachment_filename="labeled.png", mimetype="image/png")


@app.route("/api/v1/station/<string:station>/date/<string:date>/image/visual/<string:image>", methods=["GET"])
def get_visual_image(station: str, date: str, image: str):
    """
    :return: raw visual image specified by image name
    """
    base_image_name = image
    image_name = base_image_name + ".jpg"
    img = cv2.imread(join(get_visual_folder(station=station, date=date), image_name), cv2.IMREAD_COLOR)
    img_bytes = cv2.imencode(".png", img)[1]
    return send_file(io.BytesIO(img_bytes), attachment_filename="visual.png", mimetype="image/png")


@app.route("/api/v1/station/<string:station>/panel_group", methods=["GET"])
def get_panel_groups(station):
    """
    get all the panel groups positions
    :return:
    """
    db = get_mongo_client()
    cursor = db.solar.panelGroup.find(filter={"station": station}, projection={"_id": False})
    result = list()
    for post in cursor:
        result.append(post)
    return jsonify(result)


@app.route("/api/v1/station/<string:station>/panel_group", methods=["POST"])
def add_panel_group(station: str):
    """
    add a new panel group
    :return:
    """
    db = get_mongo_client()
    post = request.get_json()
    post["id"] = str(post["id"])
    if db.solar.panelGroup.find_one({"station": station, "id": post.get("id")}) is not None:
        abort(400, "naming conflict")
    db.solar.panelGroup.update_one({"id": post.get("id"), "station": station},
                                   {"$set": {"corners": post.get("corners")}}, upsert=True)
    return "OK"


@app.route("/api/v1/station/<string:station>/panel_group/<string:group_id>", methods=["GET"])
def get_panel_group(station: str, group_id: str):
    """
    get the details of a panel group
    """
    db = get_mongo_client()
    result = db.solar.panelGroup.find_one(filter={"id": group_id, "station": station}, projection={"_id": False})
    return jsonify(result)


@app.route("/api/v1/station/<string:station>/panel_group/<string:group_id>", methods=["PUT"])
def set_panel_group(station: str, group_id: str):
    """
    set the name and/or the corners of a panel group
    """
    db = get_mongo_client()
    post = request.get_json()

    if db.solar.panelGroup.find_one({"id": group_id, "station": station}) is None:
        abort(404)

    if post.get("id") is not None:
        if post.get("id") != group_id:
            current = db.solar.panelGroup.find_one({"id": post.get("id"), "station": station})
            if current is not None:
                abort(400, "naming conflict")
            else:
                db.solar.panelGroup.update_one({"id": group_id, "station": station}, {"$set": {"id": post.get("id")}})
        if post.get("corners") is not None:
            db.solar.panelGroup.update_one({"id": post.get("id"), "station": station},
                                           {"$set": {"corners": post.get("corners")}})
    elif post.get("corners") is not None:
        db.solar.panelGroup.update_one({"id": group_id, "station": station},
                                       {"$set": {"corners", post.get("corners")}})
    return "OK"


@app.route("/api/v1/station/<string:station>/date/<string:date>/image/<string:image>/temperature/point", methods=["GET"])
def get_point_temperature(station: str, date: str, image: str):
    """
    :return: temperature in celsius degree at a provided point
    """
    exif = get_exif(station=station, date=date, image=image)
    transformer = TempTransformer(e=exif.get("Emissivity"),
                                  od=exif.get("RelativeAltitude"),
                                  rtemp=22.0,
                                  atemp=22.0,
                                  irwtemp=22.0,
                                  irt=exif.get("IRWindowTransmission"),
                                  rh=50,
                                  pr1=exif.get("PlanckR1"),
                                  pb=exif.get("PlanckB"),
                                  pf=exif.get("PlanckF"),
                                  po=exif.get("PlanckO"),
                                  pr2=exif.get("PlanckR2"))
    row = int(request.args.get("row"))
    col = int(request.args.get("col"))
    raw = cv2.imread(join(get_image_root(), station, date, "ir", "rotated-raw", "{}.tif".format(image)),
                     cv2.IMREAD_ANYDEPTH)
    result = {"temperature": round(transformer.raw2temp(raw[row, col]), 1)}

    return jsonify(result)


@app.route("/api/v1/station/<string:station>/date/<string:date>/image/<string:image>/temperature/range", methods=["GET"])
def get_range_temperature(station: str, date: str, image: str):
    """
    :return: the temperature profile in an rectangle area of the image
    """
    top = int(request.args.get("top"))
    btm = int(request.args.get("btm"))
    left = int(request.args.get("left"))
    right = int(request.args.get("right"))

    exif = get_exif(station=station, date=date, image=image)
    transformer = TempTransformer(e=exif.get("Emissivity"),
                                  od=exif.get("RelativeAltitude"),
                                  rtemp=22.0,
                                  atemp=22.0,
                                  irwtemp=22.0,
                                  irt=exif.get("IRWindowTransmission"),
                                  rh=50,
                                  pr1=exif.get("PlanckR1"),
                                  pb=exif.get("PlanckB"),
                                  pf=exif.get("PlanckF"),
                                  po=exif.get("PlanckO"),
                                  pr2=exif.get("PlanckR2"))

    raw = cv2.imread(join(get_image_root(), station, date, "ir", "rotated-raw", "{}.tif".format(image)),
                     cv2.IMREAD_ANYDEPTH)
    raw_crop = raw[top: btm, left: right]
    raw_crop[raw_crop == 0] = int(raw_crop.mean())
    min_temp = transformer.raw2temp(raw_crop.min())
    max_temp = transformer.raw2temp(raw_crop.max())
    mean_temp = transformer.raw2temp(raw_crop.mean())
    min_position = raw_crop.argmin(axis=-1).data
    max_position = raw_crop.argmax(axis=-1).data

    result = {"max": round(max_temp, 1),
              "min": round(min_temp, 1),
              "mean": round(mean_temp, 1),
              "maxPos": {"row": max_position[0]+top,
                         "col": max_position[1]+left},
              "minPos": {"row": min_position[0]+top,
                         "col": min_position[1]+left}}

    return jsonify(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("log"), logging.StreamHandler()])

    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
