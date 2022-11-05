import json

from flask import Flask, request
from flask_cors import CORS

import sustainable_alternatives


app = Flask(__name__)
CORS(app)


@app.route('/get_alternatives')
def get_alternatives():
    name = request.args.get('name', '')

    # pass the name to data analysis script

    return sustainable_alternatives.get_alternatives(name)


@app.route('/get_info')
def get_info():
    name = request.args.get('name', '')

    return sustainable_alternatives.get_info(name)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
