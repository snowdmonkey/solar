from flask import Blueprint
# from flask_httpauth import HTTPBasicAuth
# from .models import User
# from ..database import db
from flask import g, jsonify
from ..misc import API_BASE
from .import auth

auth_br = Blueprint("auth", __name__)


@auth_br.route(API_BASE + "/login", methods=['POST'])
@auth.login_required
def login():
    token = g.user.generate_auth_token(600)
    return jsonify({'token': token.decode('ascii'), 'duration': 600})


@auth_br.route(API_BASE + "/logout", methods=['POST'])
@auth.login_required
def logout():
    # todo: log the user id and its logout time
    return 'Bye!', 200


@auth_br.route(API_BASE + "/refresh_token", methods=['POST'])
@auth.login_required
def get_auth_token():
    token = g.user.generate_auth_token(600)
    return jsonify({'token': token.decode('ascii'), 'duration': 600})