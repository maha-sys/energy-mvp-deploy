import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pandas as pd
from utils.preprocessing import load_encoders

# Suppress TF warnings
tf.get_logger().setLevel('ERROR')

# Project root and model paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(ROOT_DIR, 'model', 'energy_predictor.h5')

# Load model once at import if available
try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded for prediction module")
except Exception:
    MODEL = None

# Load saved encoders (scaler, month_encoder)
try:
    scaler, month_encoder = load_encoders()
except Exception as e:
    scaler = None
    month_encoder = None
    print('Encoders not found:', e)

def predict_energy(month, avg_daily, peak_hours):
    """
    Predict energy consumption based on input features
    
    Args:
        month: Month name (e.g., "Jul")
        avg_daily: Average daily kWh
        peak_hours: Peak usage hours per day
    
    Returns:
        Predicted energy consumption in kWh
    """
    # Ensure encoders are available
    if scaler is None or month_encoder is None:
        raise FileNotFoundError("❌ Encoders not found. Please train the model first using train_model.py")

    # Encode month
    try:
        month_encoded = month_encoder.transform([month])[0]
    except Exception:
        valid_months = list(getattr(month_encoder, 'classes_', []))
        raise ValueError(f"Invalid month '{month}'. Valid options: {valid_months}")

    # Prepare input as DataFrame with proper column names (fixes warning)
    input_data = pd.DataFrame(
        [[month_encoded, avg_daily, peak_hours]], 
        columns=['Month_Encoded', 'Avg_Daily_KWh', 'Peak_Usage_Hours']
    )
    input_scaled = scaler.transform(input_data)

    # Predict using loaded MODEL
    if MODEL is None:
        raise FileNotFoundError("❌ Model not found. Please train the model first using train_model.py")

    prediction = MODEL.predict(input_scaled, verbose=0)
    predicted_kwh = float(prediction[0][0])

    return predicted_kwh


def predict_with_cost(month, avg_daily, peak_hours, cost_per_kwh=6.5):
    """
    Predict energy consumption and calculate cost
    
    Args:
        month: Month name
        avg_daily: Average daily kWh
        peak_hours: Peak usage hours per day
        cost_per_kwh: Cost per kWh (default ₹6.5)
    
    Returns:
        dict with predicted_kwh and estimated_cost
    """
    predicted_kwh = predict_energy(month, avg_daily, peak_hours)
    estimated_cost = predicted_kwh * cost_per_kwh
    
    return {
        'predicted_kwh': round(predicted_kwh, 2),
        'estimated_cost': round(estimated_cost, 2),
        'cost_per_kwh': cost_per_kwh
    }


# Test prediction
if __name__ == "__main__":
    print("=" * 60)
    print("🔮 TESTING ENERGY PREDICTION")
    print("=" * 60)
    
    test_cases = [
        ("Jan", 7.0, 6, "Winter - Low Usage"),
        ("Jul", 11.6, 11, "Summer - High Usage"),
        ("May", 9.8, 9, "Spring - Medium Usage"),
        ("Dec", 8.0, 7, "Winter - Medium Usage")
    ]
    
    for month, avg_daily, peak_hours, description in test_cases:
        result = predict_with_cost(month, avg_daily, peak_hours)
        print(f"\n📅 {month} ({description})")
        print(f"   Input: {avg_daily} kWh/day, {peak_hours}h peak")
        print(f"   → Predicted: {result['predicted_kwh']} kWh")
        print(f"   → Cost: ₹{result['estimated_cost']}")
    
    print("\n" + "=" * 60)
    print("✅ Prediction test complete!")
    print("=" * 60)