import os
import shutil
from spi_app.database import get_mongo_client
from pymongo import collection
from typing import Union
from os.path import join


UPLOAD_FOLDER = '/usr/src/app/data'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
API_BASE = os.environ.get("BRAND", '') + '/api/v1'
SECRET_KEY = 'the quick brown fox jumps over the lazy dog'

image_root_path = None


def get_image_root() -> str:
    global image_root_path
    if image_root_path is None:
        image_root_path = os.getenv("IMG_ROOT")
    return image_root_path


def get_panel_group_collection() -> collection:
    return get_mongo_client().get_database("solar").get_collection("panelGroup")


def get_defect_collection() -> collection:
    return get_mongo_client().get_database("solar").get_collection("defect")


def get_exif_collection() -> collection:
    return get_mongo_client().get_database("solar").get_collection("exif")


def get_station_collection() -> collection:
    return get_mongo_client().get_database("solar").get_collection("station")


def get_log_collection() -> collection:
    return get_mongo_client().get_database("solar").get_collection("log")


def get_rect_collection() -> collection:
    return get_mongo_client().get_database("solar").get_collection("rect")


def get_exif(station: str, date: str, image: str) -> Union[dict, None]:

    exif = get_mongo_client().solar.exif.find_one({"station": station, "date": date, "image": image}, {"_id": 0})

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


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def reset_dir(station, date, image_type):
    # image_dir = os.path.join(app.config['UPLOAD_FOLDER'], station, date, image_type)
    image_dir = os.path.join(UPLOAD_FOLDER, station, date, image_type)
    # date_dir = os.path.join(app.config['UPLOAD_FOLDER'], station, date)
    date_dir = os.path.join(UPLOAD_FOLDER, station, date)
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir, ignore_errors=True)
    if not os.path.exists(date_dir):
        os.mkdir(date_dir)
    os.mkdir(image_dir)


def check_dir(station, date, image_type):
    # image_dir = os.path.join(app.config['UPLOAD_FOLDER'], station, date, image_type)
    image_dir = os.path.join(UPLOAD_FOLDER, station, date, image_type)
    # date_dir = os.path.join(app.config['UPLOAD_FOLDER'], station, date)
    date_dir = os.path.join(UPLOAD_FOLDER, station, date)
    if not os.path.exists(date_dir):
        os.mkdir(date_dir)
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)


