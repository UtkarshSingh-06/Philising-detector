from flask import Flask, request, jsonify
from src.predict import predict

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(force=True)
    features = data.get("features", [])
    if len(features) != 30:
        return jsonify({"error": "Expected 30 features"}), 400

    result = predict(features)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)

