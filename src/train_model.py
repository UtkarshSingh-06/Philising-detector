import os
import sys

# ✅ Add parent directory to Python path so it can find 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import load_and_preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt  # Added for feature importance plotting

def train_and_save_model():
    print("🔄 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

    print("🤖 Training model...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

    print("📈 Plotting feature importance...")
    importances = model.feature_importances_
    features = [f"Feature {i+1}" for i in range(len(importances))]  # Since X is scaled, use generic names

    plt.figure(figsize=(10, 6))
    plt.barh(features, importances)
    plt.xlabel("Feature Importance")
    plt.title("Phishing Detection Feature Importance")
    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/feature_importance.png")
    plt.show()

    print("💾 Saving model and scaler...")
    joblib.dump(model, 'models/phishing_rf_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✅ Done: Model and scaler saved to /models")

if __name__ == "__main__":
    train_and_save_model()

