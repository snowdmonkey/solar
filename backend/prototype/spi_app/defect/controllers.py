from flask import Blueprint, request, jsonify, abort
from ..misc import API_BASE, get_defect_collection, get_image_root, get_log_collection
from ..database import get_mongo_client
from os.path import join
from image_process import ImageProcessPipeline
from pymongo import collection
from datetime import datetime
from threading import Thread
import pymongo
import logging
import uuid


defect_br = Blueprint("defect", __name__)


@defect_br.route(API_BASE + "/station/<string:station>/date", methods=["GET"])
def get_reports_by_date_station(station: str):
    """
    return the dates of available reports for a station
    """
    coll = get_mongo_client().get_database("solar").get_collection("exif")
    dates = coll.find({"station": station}, {"date": True}).distinct(key="date")
    dates.sort()
    return jsonify(dates)


def severity2grade(severity: float) -> int:
    """
    scale severity to integers between 1 to 10
    :param severity: severity, a float greater than 4
    :return: severity grade, an integer between 1 to 10
    """
    grade = int(severity - 3)

    if grade > 10:
        grade = 10

    return grade


@defect_br.route(API_BASE + "/station/<string:station>/date/<string:date>/defects", methods=["GET"])
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
    index = 0
    for post in defects:
        defect = {
            "defectId": post.get("defectId"),
            "latitude": post.get("lat"),
            "longitude": post.get("lng"),
            "category": post.get("category"),
            "groupId": post.get("panelGroupId"),
            "severity": severity2grade(post.get("severity")),
            "index": index
        }
        results.append(defect)
        index += 1

    return jsonify(results)


class MongoHandler(logging.Handler):

    def __init__(self, job_id: str, coll: collection):

        super().__init__()
        self._job_id = job_id
        self._coll = coll

    def emit(self, record):

        d = {
            'timestamp': datetime.utcnow(),
            'level': record.levelname,
            # 'thread': record.thread,
            # 'threadName': record.threadName,
            'message': record.getMessage(),
            'loggerName': record.name,
            # 'fileName': record.pathname,
            'module': record.module,
            'method': record.funcName,
            'lineNumber': record.lineno,
            'jobId': self._job_id
        }

        self._coll.insert_one(d)


@defect_br.route(API_BASE + "/station/<string:station>/date/<string:date>/analysis", methods=["POST"])
def analyze_by_date_and_station(station: str, date: str):
    folder_path = join(get_image_root(), station, date)
    log_id = str(uuid.uuid4())

    def target_func(folder_path: str, station: str, date: str):

        logger = logging.getLogger()

        logger.addHandler(logging.FileHandler(log_id))
        logger.addHandler(MongoHandler(job_id=log_id,
                                       coll=get_mongo_client().get_database("solar").get_collection("log")))

        coll_log = get_log_collection()

        coll_log.insert_one({"jobId": log_id, "status": "running", "timestamp": datetime.utcnow()})

        try:
            pipeline = ImageProcessPipeline(image_folder=folder_path, station=station, date=date)
            pipeline.run()
            coll_log.insert_one({"jobId": log_id, "status": "completed", "timestamp": datetime.utcnow()})
        except Exception as e:
            logger.exception("message")
            coll_log.insert_one({"jobId": log_id, "status": "failed", "timestamp": datetime.utcnow()})

    t = Thread(target=target_func, args=(folder_path, station, date))
    t.start()

    return jsonify({"jobId": log_id})


@defect_br.route(API_BASE + "/station/<string:station>/date/<string:date>/defect/<string:defect_id>", methods=["PUT"])
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


@defect_br.route(API_BASE + "/station/<string:station>/date/<string:date>/defects", methods=["PUT"])
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
