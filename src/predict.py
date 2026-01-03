
import joblib
import numpy as np

# ✅ This function must exist
def predict(sample_features):
    model = joblib.load('models/phishing_rf_model.pkl')
    scaler = joblib.load('models/scaler.pkl')

    sample = np.array(sample_features).reshape(1, -1)
    scaled = scaler.transform(sample)
    prediction = model.predict(scaled)

    return "Phishing" if prediction[0] == 1 else "Legitimate"

# Optional testing from CLI
if __name__ == "__main__":
    # Use an example with 30 binary features
    features = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    print(predict(features))


