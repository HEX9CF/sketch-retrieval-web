from flask import jsonify, render_template

from app import app
from app.models.resp import Resp


@app.route('/')
def home():
    # resp = Resp(1, 'OK', None)
    # return jsonify(resp.__dict__)
    return render_template('index.html')