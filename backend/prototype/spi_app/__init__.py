"""backend app for solar panel inspection
"""
from flask import Flask
from flask_cors import CORS
from .database import db
from .misc import UPLOAD_FOLDER, SECRET_KEY
from .auth import auth_br
from .station import station_br
from .status import status_br
from .defect import defect_br
from .image import image_br
from .panel_group import panel_group_br
from .temp import temperature_br

from spi_app.auth.models import User


def create_app():
    app = Flask(__name__)
    CORS(app)

    # app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy dog'
    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sqlite/db.sqlite'
    app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    db.init_app(app)

    app.register_blueprint(auth_br)
    app.register_blueprint(station_br)
    app.register_blueprint(status_br)
    app.register_blueprint(defect_br)
    app.register_blueprint(image_br)
    app.register_blueprint(panel_group_br)
    app.register_blueprint(temperature_br)

    return app
