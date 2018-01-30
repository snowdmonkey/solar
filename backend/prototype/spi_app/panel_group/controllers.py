from flask import Blueprint, jsonify, request, abort
from ..misc import API_BASE
from ..database import get_mongo_client

panel_group_br = Blueprint("panel_group", __name__)


@panel_group_br.route(API_BASE + "/station/<string:station>/panel_group", methods=["GET"])
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


@panel_group_br.route(API_BASE + "/station/<string:station>/panel_group", methods=["POST"])
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


@panel_group_br.route(API_BASE + "/station/<string:station>/panel_group/<string:group_id>", methods=["GET"])
def get_panel_group(station: str, group_id: str):
    """
    get the details of a panel group
    """
    db = get_mongo_client()
    result = db.solar.panelGroup.find_one(filter={"id": group_id, "station": station}, projection={"_id": False})
    return jsonify(result)


@panel_group_br.route(API_BASE + "/station/<string:station>/panel_group/<string:group_id>", methods=["PUT"])
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
