"""define database used in backend
"""
from flask_sqlalchemy import SQLAlchemy
from pymongo import MongoClient
import os

db = SQLAlchemy()

mongo_client = None


def get_mongo_client() -> MongoClient:
    global mongo_client
    if mongo_client is None:
        mongo_host = os.getenv("MONGO_HOST", "mongo")
        mongo_client = MongoClient(host=mongo_host, port=27017)
    return mongo_client
