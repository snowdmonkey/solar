# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable

from datetime import datetime
from flask import Flask, request, send_file, abort, jsonify, g
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_httpauth import HTTPBasicAuth
from passlib.apps import custom_app_context as pwd_context
from os.path import join
from image_process import ImageProcessPipeline
from pymongo import MongoClient, collection
import shutil
from typing import Union, List, Optional
from temperature import TempTransformer
from itsdangerous import (TimedJSONWebSignatureSerializer
                          as Serializer, BadSignature, SignatureExpired)
from threading import Thread

try:
    from .models import *
except Exception:
    from models import *

import cv2
import io
import logging
import os
import pymongo
import uuid

UPLOAD_FOLDER = '/usr/src/app/data'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
API_BASE = os.environ.get("BRAND", '') + '/api/v1'
app = Flask(__name__)
CORS(app)
# logger = logging.getLogger(__name__)

app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy dog'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sqlite/db.sqlite'
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# extensions

db = SQLAlchemy(app)
auth = HTTPBasicAuth()

image_root_path = None
mongo_client = None


class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(32), index=True)
    password_hash = db.Column(db.String(64))

    def hash_password(self, password):
        self.password_hash = pwd_context.encrypt(password)

    def verify_password(self, password):
        return pwd_context.verify(password, self.password_hash)

    def generate_auth_token(self, expiration=600):
        s = Serializer(app.config['SECRET_KEY'], expires_in=expiration)
        return s.dumps({'id': self.id})

    @staticmethod
    def verify_auth_token(token):
        s = Serializer(app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except SignatureExpired:
            return None  # valid token, but expired
        except BadSignature:
            return None  # invalid token
        user = User.query.get(data['id'])
        return user


@auth.verify_password
def verify_password(username_or_token, password):
    # first try to authenticate by token
    user = User.verify_auth_token(username_or_token)
    if not user:
        # try to authenticate with username/password
        user = User.query.filter_by(username=username_or_token).first()
        if not user or not user.verify_password(password):
            return False
    g.user = user
    return True


def add_user():
    username = 'jason'
    password = 'Pass1234'
    if User.query.filter_by(username=username).first() is None:
        user = User(username=username)
        user.hash_password(password)
        db.session.add(user)
        db.session.commit()


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


def get_panel_group_collection() -> collection:
    return get_mongo_client().get_database("solar").get_collection("panelGroup")


def get_defect_collection() -> collection:
    return get_mongo_client().get_database("solar").get_collection("defect")


def get_exif_collection() -> collection:
    return get_mongo_client().get_database("solar").get_collection("exif")


def get_station_collection() -> collection:
    return get_mongo_client().get_database("solar").get_collection("station")


def _get_station(station_id: str) -> Optional[Station]:
    station_coll = get_station_collection()
    post = station_coll.find_one({"stationId": station_id}, {"_id": 0})

    if post is None:
        return

    station = Station(stationId=post.get("stationId"),
                      stationName=post.get("stationName"),
                      description=post.get("description"),
                      gps=tuple(post.get("gps")))
    return station


@app.route(API_BASE + "/login", methods=['POST'])
@auth.login_required
def login():
    token = g.user.generate_auth_token(600)
    return jsonify({'token': token.decode('ascii'), 'duration': 600})


@app.route(API_BASE + "/refresh_token", methods=['POST'])
@auth.login_required
def get_auth_token():
    token = g.user.generate_auth_token(600)
    return jsonify({'token': token.decode('ascii'), 'duration': 600})


@app.route(API_BASE + "/resource", methods=['GET'])
@auth.login_required
def get_resource():
    return jsonify({'data': 'Hello, %s!' % g.user.username})


@app.route(API_BASE + "/station", methods=["GET"])
def get_station_list():
    """
    :return: a list of stations
    """
    posts = get_mongo_client().solar.station.find({}, {"_id": False})
    results = [x for x in posts]
    return jsonify(results)


@app.route(API_BASE + "/station", methods=["POST"])
def add_station():
    """
    add a station
    """
    station_coll = get_mongo_client().get_database("solar").get_collection("station")
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
        station_coll.insert_one({
            "stationId": station_id,
            "stationName": station_name,
            "description": station_description,
            "gps": station_gps
        })
        return "OK"


@app.route(API_BASE + "/station/<string:station>", methods=["GET"])
def get_station_by_id(station):
    """
    return the profile of a station
    """
    result = get_mongo_client().solar.station.find_one({"stationId": station}, {"_id": False})
    if result is None:
        abort(404)
    else:
        return jsonify(result)

# API for station status


def _get_station_status(station: str, date: Optional[str] = None) -> Optional[StationStatus]:
    """
    return station status of a given date, default to the latest status
    :param station: station id
    :param date: date, default to be the latest
    :return: station status
    """
    defect_coll = get_defect_collection()
    exif_coll = get_exif_collection()

    if date is None:
        dates = exif_coll.find({"station": station}, {"_id": 0, "date": 1}).distinct("date")
        dates.sort()

        if len(dates) == 0:
            return

        date = dates[-1]

    n_to_confirm = defect_coll.find({"station": station, "date": date, "category": 0}).count()
    n_confirmed = defect_coll.find({"station": station, "date": date, "category": 1}).count()
    n_healthy = defect_coll.find({"station": station, "date": date, "category": 2}).count()
    n_in_fix = defect_coll.find({"station": station, "date": date, "category": 3}).count()

    return StationStatus(date=date, healthy=n_healthy, toconfirm=n_to_confirm, infix=n_in_fix, confirmed=n_confirmed)


@app.route(API_BASE + "/status", methods=["GET"])
def get_station_status():
    """
    :return: status of all available stations
    """
    station_ids = get_station_collection().find({}, {"stationId": 1, "_id": 0}).distinct("stationId")

    results = list()

    for station_id in station_ids:
        station = _get_station(station_id)
        status = _get_station_status(station_id)

        if station is not None:
            result = {"station": station._asdict()}

            if status is None:
                result.update({"status": dict()})
            else:
                result.update({"status": status._asdict()})

            results.append(result)

        # if (station is not None) and (status is not None):
        #     results.append({"station": station._asdict(),
        #                     "status": status._asdict()})

    return jsonify(results)


@app.route(API_BASE + "/station/<string:station>/date/<string:date>/status", methods=["GET"])
def get_status_by_station_and_date(station: str, date: str):
    status = _get_station_status(station, date)
    if status is None:
        abort(404)
    return jsonify(status._asdict())


@app.route(API_BASE + "/station/<string:station>/status/start/<string:start>/end/<string:end>", methods=["GET"])
def get_status_by_station_and_range(station: str, start: str, end: str):
    # defect_coll = get_defect_collection()
    exif_coll = get_exif_collection()
    dates = exif_coll.find({"station": station}, {"_id": 0, "date": 1}).distinct("date")
    dates = [x for x in dates if start <= x <= end]
    dates.sort()
    results = list()
    for date in dates:
        status = _get_station_status(station, date)
        if status is not None:
            results.append(status._asdict())

    return jsonify(results)


@app.route(API_BASE + "/station/<string:station>/status/start/<string:start>", methods=["GET"])
def get_status_by_station_and_start(station: str, start: str):
    # defect_coll = get_defect_collection()
    exif_coll = get_exif_collection()
    dates = exif_coll.find({"station": station}, {"_id": 0, "date": 1}).distinct("date")
    dates = [x for x in dates if start <= x]
    dates.sort()
    results = list()
    for date in dates:
        status = _get_station_status(station, date)
        if status is not None:
            results.append(status._asdict())

    return jsonify(results)


@app.route(API_BASE + "/station/<string:station>/date", methods=["GET"])
def get_reports_by_date_station(station: str):
    """
    return the dates of available reports for a station
    """
    coll = get_mongo_client().get_database("solar").get_collection("exif")
    posts = coll.find({"station": station}, {"date": True}).distinct(key="date")
    return jsonify(posts)


@app.route(API_BASE + "/station/<string:station>/date/<string:date>/defects", methods=["GET"])
def get_defects_by_date_and_station(station: str, date: str):
    """
    return a defect list by station and date
    """
    # defect_summary = get_defects_summary(station, date)

    cat_str = request.args.get("category")
    defect_coll = get_defect_collection()

    if cat_str is None:
        defects = defect_coll.find({"station": station, "date": date}, {"_id": False})
    else:
        categories = [int(x) for x in cat_str.split(",")]
        defects = defect_coll.find({"station": station, "date": date, "category": {"$in": categories}}, {"_id": False})

    sort_key = request.args.get("sortby")
    if sort_key is not None:
        if sort_key in ("category", "severity"):
            sort_order = request.args.get("order")
            if sort_order is None:
                defects.sort(sort_key)
            elif sort_order == "reverse":
                defects.sort(sort_key, pymongo.DESCENDING)
            else:
                abort(400, "unknown order")
        else:
            abort(400, "unknown sort key")

    results = list()
    for post in defects:
        defect = {
            "defectId": post.get("defectId"),
            "latitude": post.get("lat"),
            "longitude": post.get("lng"),
            "category": post.get("category"),
            "groupId": post.get("panelGroupId"),
            "severity": round(post.get("severity"), 2)
        }
        results.append(defect)

    return jsonify(results)


@app.route(API_BASE + "/station/<string:station>/date/<string:date>/analysis", methods=["POST"])
def analyze_by_date_and_station(station: str, date: str):
    folder_path = join(get_image_root(), station, date)

    def target_func(folder_path: str, station: str, date: str):

        # logger = logging.getLogger()
        # log_id = str(uuid.uuid4())
        # logger.addHandler(logging.FileHandler(log_id))

        pipeline = ImageProcessPipeline(image_folder=folder_path, station=station, date=date)
        pipeline.run()

    t = Thread(target=target_func, args=(folder_path, station, date))
    t.start()
    return "OK"


@app.route(API_BASE + "/station/<string:station>/date/<string:date>/defect/<string:defect_id>", methods=["PUT"])
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
                                     {"$set": {"category": cat}})
    return "OK"


@app.route(API_BASE + "/station/<string:station>/date/<string:date>/defects", methods=["PUT"])
def set_defects(station: str, date: str):
    """
    set defects' category by batch
    """
    post_body = request.get_json()
    ids = post_body.get("ids")
    category = post_body.get("category")

    defect_coll = get_defect_collection()
    for defect_id in ids:
        defect_coll.update({"station": station, "date": date, "defectId": defect_id},
                           {"$set": {"category": category}})
    return "OK"


@app.route(API_BASE +
           "/station/<string:station>/date/<string:date>/defect/<string:defect_id>/images/ir", methods=["GET"])
def get_ir_images_by_defect(station: str, date: str, defect_id: str):
    """
    return a json string which contains the details of the images relating to a defect
    :return: json string
    """
    defect_info = get_defect_collection().find_one({"station": station, "date": date, "defectId": defect_id})
    # defect_info = get_mongo_client().get_database("solar")\
    #     .get_collection("defect")\
    #     .find_one({"station": station, "date": date, "defectId": defect_id})
    image_names = {x.get("image") for x in defect_info.get("rects")}
    color_map = request.args.get("colorMap")
    if color_map is not None:
        if color_map not in ("autumn", "bone", "jet", "winter", "rainbow", "ocean",
                             "summer", "spring", "cool", "hsv", "pink", "hot"):
            abort(400, "unknown color map")

    results = list()

    for image_name in image_names:
        exif = get_exif(station=station, date=date, image=image_name)
        lat = exif.get("GPSLatitude")
        lng = exif.get("GPSLongitude")
        image_url = API_BASE + "/station/{}/date/{}/image/ir/{}?defect={}".format(station, date, image_name, defect_id)
        if color_map is not None:
            image_url += "&colorMap={}".format(color_map)
        results.append({"imageName": image_name, "latitude": lat, "longitude": lng, "url": image_url})
    return jsonify(results)


@app.route(API_BASE + "/station/<string:station>/date/<string:date>/defect/<string:defect_id>/images/ir", methods=["GET"])
def get_visual_images_by_defect(station: str, date: str, defect_id: str):
    """
    return a json string that contains the details of visual images relating to a defect
    """
    # TODO implement how to get a visual image with the same scope of an ir image
    pass


@app.route(API_BASE + "/station/<string:station>/date/<string:date>/image/ir/<string:image>", methods=["GET"])
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


@app.route(API_BASE + "/station/<string:station>/date/<string:date>/image/visual/<string:image>", methods=["GET"])
def get_visual_image(station: str, date: str, image: str):
    """
    :return: raw visual image specified by image name
    """
    base_image_name = image
    image_name = base_image_name + ".jpg"
    img = cv2.imread(join(get_visual_folder(station=station, date=date), image_name), cv2.IMREAD_COLOR)
    img_bytes = cv2.imencode(".png", img)[1]
    return send_file(io.BytesIO(img_bytes), attachment_filename="visual.png", mimetype="image/png")


@app.route(API_BASE + "/station/<string:station>/panel_group", methods=["GET"])
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


@app.route(API_BASE + "/station/<string:station>/panel_group", methods=["POST"])
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


@app.route(API_BASE + "/station/<string:station>/panel_group/<string:group_id>", methods=["GET"])
def get_panel_group(station: str, group_id: str):
    """
    get the details of a panel group
    """
    db = get_mongo_client()
    result = db.solar.panelGroup.find_one(filter={"id": group_id, "station": station}, projection={"_id": False})
    return jsonify(result)


@app.route(API_BASE + "/station/<string:station>/panel_group/<string:group_id>", methods=["PUT"])
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


@app.route(API_BASE +
           "/station/<string:station>/date/<string:date>/image/<string:image>/temperature/point", methods=["GET"])
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
    row_ratio = float(request.args.get("row"))
    col_ratio = float(request.args.get("col"))
    raw = cv2.imread(join(get_image_root(), station, date, "ir", "rotated-raw", "{}.tif".format(image)),
                     cv2.IMREAD_ANYDEPTH)
    n_row, n_col = raw.shape
    row = int(row_ratio * n_row)
    col = int(col_ratio * n_col)

    result = {"temperature": round(transformer.raw2temp(raw[row, col]), 1)}

    return jsonify(result)


@app.route(API_BASE +
           "/station/<string:station>/date/<string:date>/image/<string:image>/temperature/range", methods=["GET"])
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


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def reset_dir(station, date, image_type):
    image_dir = os.path.join(app.config['UPLOAD_FOLDER'], station, date, image_type)
    date_dir = os.path.join(app.config['UPLOAD_FOLDER'], station, date)
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir, ignore_errors=True)
    if not os.path.exists(date_dir):
        os.mkdir(date_dir)
    os.mkdir(image_dir)


def check_dir(station, date, image_type):
    image_dir = os.path.join(app.config['UPLOAD_FOLDER'], station, date, image_type)
    date_dir = os.path.join(app.config['UPLOAD_FOLDER'], station, date)
    if not os.path.exists(date_dir):
        os.mkdir(date_dir)
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)


@app.route(API_BASE + "/station/<string:station>/date/<string:date>/image/ir", methods=['POST'])
def upload_ir_file(station, date):
    if request.method == 'POST':
        if 'file' not in request.files:
            abort(400)
        file = request.files['file']
        if file.filename == '':
            abort(400)
        if file:
            if allowed_file(file.filename):
                filename = datetime.now().isoformat()[11:].replace(':', '-') + '.' + file.filename.rsplit('.', 1)[
                    1].lower()
                check_dir(station, date, 'ir')
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], station, date, 'ir', filename))
                return 'success', 200
            else:
                return jsonify(dict(message='请选择JPG或JEPG格式的文件。')), 400
        else:
            return jsonify(dict(message='未知文件。')), 400
    abort(400)


@app.route(API_BASE + "/station/<string:station>/date/<string:date>/image/visual", methods=['POST'])
def upload_visual_file(station, date):
    if request.method == 'POST':
        if 'file' not in request.files:
            abort(400)
        file = request.files['file']
        if file.filename == '':
            abort(400)
        if file:
            if allowed_file(file.filename):
                filename = datetime.now().isoformat()[11:].replace(':', '-') + '.' + file.filename.rsplit('.', 1)[
                    1].lower()
                check_dir(station, date, 'visual')
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], station, date, 'visual', filename))
                return 'success', 200
            else:
                return jsonify(dict(message='请选择JPG或JEPG格式的文件。')), 400
        else:
            return jsonify(dict(message='未知文件。')), 400
    abort(400)


@app.route(API_BASE + "/station/<string:station>/date/<string:date>/image/el", methods=['POST'])
def upload_el_file(station, date):
    if request.method == 'POST':
        if 'file' not in request.files:
            abort(400)
        file = request.files['file']
        if file.filename == '':
            abort(400)
        if file:
            if allowed_file(file.filename):
                filename = datetime.now().isoformat()[11:].replace(':', '-') + '.' + file.filename.rsplit('.', 1)[
                    1].lower()
                check_dir(station, date, 'el')
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], station, date, 'el', filename))
                return 'success', 200
            else:
                return jsonify(dict(message='请选择JPG或JEPG格式的文件。')), 400
        else:
            return jsonify(dict(message='未知文件。')), 400
    abort(400)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("log"), logging.StreamHandler()])

    if not os.path.exists('sqlite/db.sqlite'):
        db.create_all()
        add_user()
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
