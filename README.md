# Phishing Detector

A machine learning-based phishing URL detection system that analyzes URLs using various features to classify them as legitimate or phishing.

## Features

- **Machine Learning Models**: Implements Random Forest classifier for phishing detection
- **Deep Learning Option**: Includes CNN-LSTM model for advanced analysis
- **REST API**: Flask-based API for real-time predictions
- **Preprocessing**: Automated data preprocessing and feature scaling
- **Model Training**: Scripts to train and evaluate models
- **Dataset**: Uses UCI Phishing Websites dataset with 30 features

## Project Structure

```
philsing-detector/
├── app.py                 # Flask API for predictions
├── import.py              # Utility imports
├── test_request.py        # API testing script
├── README.md              # Project documentation
├── data/
│   ├── phishing.csv       # UCI Phishing Websites dataset
│   └── checkcol.py        # Data validation script
├── dl/
│   ├── cnn_lstm_model.py  # CNN-LSTM deep learning model
│   ├── dl_model.py        # Additional DL utilities
│   └── tempCodeRunnerFile.py
├── models/
│   └── phishing_dnn_model.h5  # Trained deep learning model
└── src/
    ├── __init__.py
    ├── predict.py          # Prediction functions
    ├── preprocess.py       # Data preprocessing
    └── train_model.py      # Model training script
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd philsing-detector
```

2. Install dependencies:
```bash
pip install flask scikit-learn tensorflow pandas numpy matplotlib joblib
```

## Usage

### Training the Model

Run the training script to train the Random Forest model:

```bash
python src/train_model.py
```

This will:
- Load and preprocess the data
- Train the model
- Save the model and scaler to `models/`
- Generate feature importance plot

### Running the API

Start the Flask API server:

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Making Predictions

Send a POST request to `/predict` with 30 features:

```python
import requests

features = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
response = requests.post('http://localhost:5000/predict', json={'features': features})
print(response.json())
```

### Testing

Use the test script to verify the API:

```bash
python test_request.py
```

## Dataset

The project uses the UCI Phishing Websites dataset containing 11,055 URLs with 30 features each:

- `having_IP_Address`
- `URL_Length`
- `Shortining_Service`
- `having_At_Symbol`
- And 26 other features...

Labels:
- `-1`: Legitimate
- `1`: Phishing

## Models

### Random Forest Model
- Located in `models/phishing_rf_model.pkl`
- Trained with 200 estimators
- Includes feature scaling

### Deep Learning Model
- CNN-LSTM architecture
- Saved as `models/phishing_dnn_model.h5`
- Requires TensorFlow/Keras

## API Endpoints

- `POST /predict`: Accepts JSON with `features` array (30 elements)
  - Returns: `{"result": "Phishing"}` or `{"result": "Legitimate"}`

## Requirements

- Python 3.7+
- Flask
- scikit-learn
- TensorFlow 2.x
- pandas
- numpy
- matplotlib
- joblib

## License

This project is open source. Please check the license file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

- UCI Machine Learning Repository for the phishing dataset
- Scikit-learn and TensorFlow communities