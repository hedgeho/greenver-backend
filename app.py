import json

from flask import Flask, request
from flask_cors import CORS

import sustainable_alternatives


app = Flask(__name__)
CORS(app)


@app.route('/get_alternatives')
def get_alternatives():
    id = request.args.get('id', '')
    
    return sustainable_alternatives.get_alternatives(id)


@app.route('/get_info')
def get_info():
    id = request.args.get('id', '')

    return sustainable_alternatives.get_info(id)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
