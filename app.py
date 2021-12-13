from flask import Flask, jsonify, request, Blueprint
from site_apis.price_api import prediction_api


app = Flask(__name__)
app.register_blueprint(prediction_api)


@app.route('/')
@app.route('/home')
def home():
    # testing, testing...
    return jsonify(healthcheck='ok')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
