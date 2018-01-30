from flask import Blueprint, request, jsonify
from spi_app.misc import API_BASE, get_exif, get_image_root
from os.path import join
from temperature import TempTransformer
import cv2

temperature_br = Blueprint("temp", __name__)


@temperature_br.route(API_BASE +
                      "/station/<string:station>/date/<string:date>/image/<string:image>/temp/point",
                      methods=["GET"])
def get_point_temperature(station: str, date: str, image: str):
    """
    :return: temp in celsius degree at a provided point
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

    result = {"temp": round(transformer.raw2temp(raw[row, col]), 1)}

    return jsonify(result)


@temperature_br.route(API_BASE +
                      "/station/<string:station>/date/<string:date>/image/<string:image>/temp/range",
                      methods=["GET"])
def get_range_temperature(station: str, date: str, image: str):
    """
    :return: the temp profile in an rectangle area of the image
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
              "maxPos": {"row": max_position[0] + top,
                         "col": max_position[1] + left},
              "minPos": {"row": min_position[0] + top,
                         "col": min_position[1] + left}}

    return jsonify(result)
