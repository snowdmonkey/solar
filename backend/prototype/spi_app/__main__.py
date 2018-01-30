import torch
from torch.autograd import Variable
import logging
import os
from . import app, db
from .auth.models import User


def add_user():
    username = 'jason'
    password = 'Pass1234'
    if User.query.filter_by(username=username).first() is None:
        user = User(username=username)
        user.hash_password(password)
        db.session.add(user)
        db.session.commit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("log"), logging.StreamHandler()])
    if not os.path.exists('sqlite/db.sqlite'):
        db.create_all()
        add_user()
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)