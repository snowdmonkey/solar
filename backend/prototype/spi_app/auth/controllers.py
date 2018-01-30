from flask import Blueprint
from flask_httpauth import HTTPBasicAuth
from .models import User
from ..database import db
from flask import g, jsonify
from ..misc import API_BASE

auth_br = Blueprint("auth", __name__)

auth = HTTPBasicAuth()


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