
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import numpy as np
from utils.preprocessing import load_and_preprocess
from sklearn.model_selection import train_test_split

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

print("=" * 60)
print("🚀 ENERGY CONSUMPTION PREDICTION MODEL TRAINING")
print("=" * 60)

# Load and preprocess data
X, y, scaler, month_encoder = load_and_preprocess("data/sample_energy_data.csv")

print(f"\n📊 Dataset Information:")
print(f"   - Total samples: {X.shape[0]}")
print(f"   - Features: {X.shape[1]}")
print(f"   - Target range: {y.min():.1f} - {y.max():.1f} kWh")

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\n🔀 Data Split:")
print(f"   - Training samples: {X_train.shape[0]}")
print(f"   - Validation samples: {X_val.shape[0]}")

# Build AI Regression Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae', 'mse']
)

print("\n🏗️ Model Architecture:")
model.summary()

# Early stopping to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint to save best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'model/energy_predictor.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

print("\n" + "=" * 60)
print("🚀 Starting Training...")
print("=" * 60)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=4,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Evaluate on validation set
print("\n" + "=" * 60)
val_loss, val_mae, val_mse = model.evaluate(X_val, y_val, verbose=0)
print(f"📈 Final Validation Results:")
print(f"   - MAE (Mean Absolute Error): {val_mae:.2f} kWh")
print(f"   - MSE (Mean Squared Error): {val_mse:.2f}")
print(f"   - RMSE (Root Mean Squared Error): {np.sqrt(val_mse):.2f} kWh")
print(f"   - Error Rate: {(val_mae/y.mean())*100:.2f}%")
print("=" * 60)

# Final save
model.save("model/energy_predictor.h5")

print("\n✅ AI model trained and saved successfully!")
print(f"💾 Model saved to: model/energy_predictor.h5")
print(f"📁 Encoders saved to: model/scaler.pkl & model/month_encoder.pkl")