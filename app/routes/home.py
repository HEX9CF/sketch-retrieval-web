from flask import jsonify

from app import app
from app.models.resp import Resp


@app.route('/')
def home():
    resp = Resp(1, 'OK', None)
    return jsonify(resp.__dict__)