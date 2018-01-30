from flask import Blueprint, request, abort, jsonify
from ..misc import API_BASE
from ..database import get_mongo_client
from typing import Optional
from .models import Station
from ..misc import get_station_collection

station_br = Blueprint("station", __name__)


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

@station_br.route(API_BASE + "/station", methods=["GET"])
def get_station_list():
    """
    :return: a list of stations
    """
    posts = get_mongo_client().solar.station.find({}, {"_id": False})
    results = [x for x in posts]
    return jsonify(results)


@station_br.route(API_BASE + "/station", methods=["POST"])
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


@station_br.route(API_BASE + "/station/<string:station>", methods=["GET"])
def get_station_by_id(station):
    """
    return the profile of a station
    """
    result = get_mongo_client().solar.station.find_one({"stationId": station}, {"_id": False})
    if result is None:
        abort(404)
    else:
        return jsonify(result)
