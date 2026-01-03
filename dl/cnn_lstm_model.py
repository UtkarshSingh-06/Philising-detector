import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Flatten, Dropout
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/phishing.csv")

# Check if 'Result' column exists
if 'Result' not in df.columns:
    raise ValueError("❌ 'Result' column is required!")

# Separate features and labels
X = df.drop(columns=['Result']).values
y = df['Result'].replace(-1, 0).values  # Replace -1 with 0 for binary classification

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape input for CNN-LSTM: (samples, timesteps, features)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, stratify=y, random_state=42)

# Build CNN-LSTM model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    LSTM(64, return_sequences=True),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Final Test Accuracy: {accuracy:.4f}")

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# import matplotlib.pyplot as plt

# # Load dataset
# df = pd.read_csv("data/phishing.csv")

# # Confirm 'Result' is in dataset
# if 'Result' not in df.columns:
#     raise ValueError("❌ 'Result' column not found in the dataset!")

# # Split features and labels
# X = df.drop('Result', axis=1)
# y = df['Result'].replace(-1, 0)  # Convert -1 → 0 for binary classification

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Feature scaling
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Build a simple DNN model
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
#     Dropout(0.3),
#     Dense(32, activation='relu'),
#     Dropout(0.3),
#     Dense(1, activation='sigmoid')
# ])

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()

# # Train model and capture history
# history = model.fit(
#     X_train_scaled, y_train,
#     epochs=20,
#     batch_size=32,
#     validation_split=0.2,
#     callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
# )

# # Plot accuracy and loss
# plt.figure(figsize=(12, 5))

# # Accuracy plot
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
# plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
# plt.title('Model Accuracy Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)

# # Loss plot
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss', marker='o')
# plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
# plt.title('Model Loss Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig("models/training_plot.png")
# plt.show()

# # Evaluate
# loss, acc = model.evaluate(X_test_scaled, y_test)
# print(f"\n✅ Test Accuracy: {acc:.4f}")

# # Save model and scaler
# model.save("models/phishing_dnn_model.h5")
# import pickle
# with open("models/scaler.pkl", "wb") as f:
#     pickle.dump(scaler, f)
# print("✅ Model and scaler saved.")
