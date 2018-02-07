from flask import Blueprint, abort, jsonify, send_file, request
from ..misc import API_BASE, get_defect_collection, get_exif, get_rotated_folder, get_visual_folder, check_dir, \
    allowed_file, UPLOAD_FOLDER, get_rect_collection
from os.path import join
from ..database import get_mongo_client
from datetime import datetime
import cv2
import io
import os


image_br = Blueprint("image", __name__)


@image_br.route(API_BASE +
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
    if defect_info is None:
        abort(404)
    image_names = {x.get("image") for x in defect_info.get("rects")}
    color_map = request.args.get("colorMap")
    if color_map is not None:
        if color_map not in ("autumn", "bone", "jet", "winter", "rainbow", "ocean",
                             "summer", "spring", "cool", "hsv", "pink", "hot"):
            abort(400, "unknown color map")

    rect_coll = get_rect_collection()

    results = list()

    for image_name in image_names:
        exif = get_exif(station=station, date=date, image=image_name)
        lat = exif.get("GPSLatitude")
        lng = exif.get("GPSLongitude")
        image_url = API_BASE + "/station/{}/date/{}/image/ir/{}?defect={}".format(station, date, image_name, defect_id)
        if color_map is not None:
            image_url += "&colorMap={}".format(color_map)

        rect = rect_coll.find_one({"station": station, "date": date, "image": image_name},
                                  {"_id": 0, "width": 1, "height": 1})
        results.append({"imageName": image_name,
                        "latitude": lat,
                        "longitude": lng,
                        "url": image_url,
                        "width": rect.get("width"),
                        "height": rect.get("height")})
    return jsonify(results)


@image_br.route(API_BASE +
                "/station/<string:station>/date/<string:date>/defect/<string:defect_id>/images/ir", methods=["GET"])
def get_visual_images_by_defect(station: str, date: str, defect_id: str):
    """
    return a json string that contains the details of visual images relating to a defect
    """
    # TODO implement how to get a visual image with the same scope of an ir image
    pass


@image_br.route(API_BASE + "/station/<string:station>/date/<string:date>/image/ir/<string:image>", methods=["GET"])
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
        rects = get_mongo_client().get_database("solar").get_collection("defect") \
            .find_one({"station": station, "date": date, "defectId": defect_id}, {"_id": 0, "rects": 1}).get("rects")
        for rect in rects:
            if rect.get("image") == image:
                x, y, w, h = rect.get("x"), rect.get("y"), rect.get("w"), rect.get("h")
                cv2.rectangle(img, (x, y), (x + w, y + h), rect_color, 1)

    img_bytes = cv2.imencode(".png", img)[1]
    return send_file(io.BytesIO(img_bytes), attachment_filename="labeled.png", mimetype="image/png")


@image_br.route(API_BASE + "/station/<string:station>/date/<string:date>/image/visual/<string:image>", methods=["GET"])
def get_visual_image(station: str, date: str, image: str):
    """
    :return: raw visual image specified by image name
    """
    base_image_name = image
    image_name = base_image_name + ".jpg"
    img = cv2.imread(join(get_visual_folder(station=station, date=date), image_name), cv2.IMREAD_COLOR)
    img_bytes = cv2.imencode(".png", img)[1]
    return send_file(io.BytesIO(img_bytes), attachment_filename="visual.png", mimetype="image/png")


@image_br.route(API_BASE + "/station/<string:station>/date/<string:date>/image/ir", methods=['POST'])
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
                # file.save(os.path.join(app.config['UPLOAD_FOLDER'], station, date, 'ir', filename))
                file.save(os.path.join(UPLOAD_FOLDER, station, date, 'ir', filename))
                return 'success', 200
            else:
                return jsonify(dict(message='请选择JPG或JEPG格式的文件。')), 400
        else:
            return jsonify(dict(message='未知文件。')), 400
    abort(400)


@image_br.route(API_BASE + "/station/<string:station>/date/<string:date>/image/visual", methods=['POST'])
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
                # file.save(os.path.join(app.config['UPLOAD_FOLDER'], station, date, 'visual', filename))
                file.save(os.path.join(UPLOAD_FOLDER, station, date, 'visual', filename))
                return 'success', 200
            else:
                return jsonify(dict(message='请选择JPG或JEPG格式的文件。')), 400
        else:
            return jsonify(dict(message='未知文件。')), 400
    abort(400)


@image_br.route(API_BASE + "/station/<string:station>/date/<string:date>/image/el", methods=['POST'])
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
                # file.save(os.path.join(app.config['UPLOAD_FOLDER'], station, date, 'el', filename))
                file.save(os.path.join(UPLOAD_FOLDER, station, date, 'el', filename))
                return 'success', 200
            else:
                return jsonify(dict(message='请选择JPG或JEPG格式的文件。')), 400
        else:
            return jsonify(dict(message='未知文件。')), 400
    abort(400)
