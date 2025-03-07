from flask import Flask, request, jsonify
import requests

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    return jsonify(response.json())


if __name__ == "__main__":
    app.run(debug=True, port=5000)
