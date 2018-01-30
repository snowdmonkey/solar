"""API for station status
"""
from flask import Blueprint, jsonify, abort, request
from ..misc import API_BASE, get_image_root, get_panel_group_collection, get_exif_collection, get_defect_collection, \
    get_station_collection
from .models import StationStatus
from typing import Optional, List
from ..station.controllers import _get_station
import os
import json


status_br = Blueprint("status", __name__)


def _get_n_panel_group(station_id: str) -> int:
    """
    get the number of panel groups in a station
    :param station_id: station id
    :return: number of panel groups in a station
    """
    panel_group_coll = get_panel_group_collection()
    n_panel_group = panel_group_coll.find({"station": station_id}).count()

    if n_panel_group == 0:
        file_path = os.path.join(get_image_root(), station_id, "groupPanel.json")

        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                panel_groups = json.load(f)

            for d in panel_groups:
                d.update({"station": station_id})

            panel_group_coll.insert_many(panel_groups)
            return len(panel_groups)
        else:
            return 0
    else:
        return n_panel_group


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

    n_panel_groups = _get_n_panel_group(station)

    if n_panel_groups == 0:
        overall_status = "unknown"
    else:
        problem_ratio = (n_to_confirm+n_confirmed+n_in_fix) / n_panel_groups
        if problem_ratio > 0.5:
            overall_status = "red"
        elif problem_ratio > 0.05:
            overall_status = "yellow"
        else:
            overall_status = "green"

    return StationStatus(date=date,
                         healthy=n_healthy,
                         toconfirm=n_to_confirm,
                         infix=n_in_fix,
                         confirmed=n_confirmed,
                         overallStatus=overall_status)


@status_br.route(API_BASE + "/status", methods=["GET"])
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


@status_br.route(API_BASE + "/station/<string:station>/date/<string:date>/status", methods=["GET"])
def get_status_by_station_and_date(station: str, date: str):
    status = _get_station_status(station, date)
    if status is None:
        abort(404)
    return jsonify(status._asdict())


@status_br.route(API_BASE + "/station/<string:station>/status/start/<string:start>/end/<string:end>", methods=["GET"])
def get_status_by_station_and_range(station: str, start: str, end: str):
    # defect_coll = get_defect_collection()
    exif_coll = get_exif_collection()
    dates = exif_coll.find({"station": station}, {"_id": 0, "date": 1}).distinct("date")
    dates = [x for x in dates if start <= x <= end]

    results = list()  # type: List[StationStatus]
    for date in dates:
        status = _get_station_status(station, date)
        if status is not None:
            results.append(status)

    sort_key = request.args.get("sortby", "date")

    if sort_key == "date":
        results.sort(key=lambda x: x.date)
    elif sort_key == "frate":
        results.sort(key=lambda x: x.confirmed)
    else:
        abort(400, "unsupported sort key")

    order = request.args.get("order", "descending")

    if order == "ascending":
        pass
    elif order == "descending":
        results.reverse()
    else:
        abort(400, "unsupported sort order")

    results = [x._asdict() for x in results]

    return jsonify(results)


@status_br.route(API_BASE + "/station/<string:station>/status/start/<string:start>", methods=["GET"])
def get_status_by_station_and_start(station: str, start: str):
    # defect_coll = get_defect_collection()
    exif_coll = get_exif_collection()
    dates = exif_coll.find({"station": station}, {"_id": 0, "date": 1}).distinct("date")
    dates = [x for x in dates if start <= x]

    results = list()  # type: List[StationStatus]
    for date in dates:
        status = _get_station_status(station, date)
        if status is not None:
            results.append(status)

    sort_key = request.args.get("sortby", "date")

    if sort_key == "date":
        results.sort(key=lambda x: x.date)
    elif sort_key == "frate":
        results.sort(key=lambda x: x.confirmed)
    else:
        abort(400, "unsupported sort key")

    order = request.args.get("order", "descending")

    if order == "ascending":
        pass
    elif order == "descending":
        results.reverse()
    else:
        abort(400, "unsupported sort order")

    results = [x._asdict() for x in results]

    return jsonify(results)
