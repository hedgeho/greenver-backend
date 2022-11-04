from flask import Flask, request

app = Flask(__name__)


@app.route('/get_alternatives')
def hello_world():
    name = request.args.get('name', '')

    # pass the name to data analysis script
    

    return 'Hello World!'


if __name__ == '__main__':
    app.run()
