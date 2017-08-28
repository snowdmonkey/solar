from __future__ import print_function

import base64
import os
import json
import tempfile
import random

import matplotlib
matplotlib.use('Agg')

from flask import Flask, request
from flask_restful import Resource, Api

from image_process import process


app = Flask(__name__)
api = Api(app)


class Echo(Resource):
    def get(self):
        return 'echo'


def getImage(req):
    ret = ''
    with tempfile.TemporaryDirectory() as tmpdir:
        input = os.sep.join((tmpdir, str(random.randint(100, 10000)))) + '.jpg'
        output = os.sep.join((tmpdir, str(random.randint(100, 10000)))) + '.png'

        ifile = open(input, 'wb')
        ifile.write(base64.b64decode(req.get('image')))
        ifile.flush()
        ifile.close()
        result = process(input, output)

        if result:
            res = open(output, 'rb')
            ret = base64.b64encode(res.read())
            res.close()
    return result, ret


class Spot(Resource):
    def post(self):
        req = request.get_json()
        headerkv = {'Content-Type': 'application/json'}
        result, img = getImage(req)
        if result:
            body = '{"detected": %s, "image": "%s"}' % ('true', img)
        else:
            body = '{"detected": false, "image": ""}'
        resp = api.make_response(json.loads(body), 200, headers=headerkv)
        return resp


api.add_resource(Echo, '/echo')
api.add_resource(Spot, '/api/v1/images/spotdetect')

if __name__ == '__main__':
    if 'DEBUG' in os.environ:
        mode = os.environ['DEBUG'] == 'TRUE'
    else:
        mode = False
    app.run(host='0.0.0.0', port=8080, debug=mode)