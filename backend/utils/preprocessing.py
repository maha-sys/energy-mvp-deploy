
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pickle
import os

def load_and_preprocess(csv_path, save_encoders=True):
    """
    Load and preprocess energy data
    
    Args:
        csv_path: Path to CSV file
        save_encoders: Whether to save scaler and encoder for later use
    
    Returns:
        X_scaled: Scaled feature matrix
        y: Target variable (Units_kWh)
        scaler: Fitted MinMaxScaler
        month_encoder: Fitted LabelEncoder
    """
    
    # Load data with error handling
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ CSV file not found: {csv_path}")
    except Exception as e:
        raise Exception(f"❌ Error loading CSV: {str(e)}")
    
    # Extract just the month name (remove year if present)
    # "Jan-2020" -> "Jan", "Feb" -> "Feb"
    df['Month'] = df['Month'].str.split('-').str[0]
    
    # Validate required columns
    required_cols = ['Month', 'Avg_Daily_KWh', 'Peak_Usage_Hours', 'Units_kWh']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing required columns: {missing_cols}")
    
    # Check for missing values
    if df[required_cols].isnull().any().any():
        print("⚠️  Warning: Missing values detected. Filling with median...")
        df[required_cols] = df[required_cols].fillna(df[required_cols].median())
    
    # Encode Month (Jan, Feb, etc.)
    month_encoder = LabelEncoder()
    df['Month_Encoded'] = month_encoder.fit_transform(df['Month'])
    
    # Features and target (removed 'Cost' to avoid data leakage)
    # Create DataFrame with proper column names to avoid warning
    X = pd.DataFrame({
        'Month_Encoded': df['Month_Encoded'],
        'Avg_Daily_KWh': df['Avg_Daily_KWh'],
        'Peak_Usage_Hours': df['Peak_Usage_Hours']
    })
    y = df['Units_kWh'].values
    
    # Scale features - MinMaxScaler will now preserve column names
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save encoders for prediction phase to the project's model directory
    if save_encoders:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_dir = os.path.join(root_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join(model_dir, 'month_encoder.pkl'), 'wb') as f:
            pickle.dump(month_encoder, f)
        print(f"💾 Scaler and encoder saved to {model_dir}")
    
    return X_scaled, y, scaler, month_encoder


def load_encoders():
    """
    Load saved scaler and encoder for prediction
    
    Returns:
        scaler: Fitted MinMaxScaler
        month_encoder: Fitted LabelEncoder
    """
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_dir = os.path.join(root_dir, 'model')
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(model_dir, 'month_encoder.pkl'), 'rb') as f:
            month_encoder = pickle.load(f)
        return scaler, month_encoder
    except FileNotFoundError:
        raise FileNotFoundError(
            "❌ Encoders not found. Please train the model first using train_model.py"
        )


if __name__ == "__main__":
    # Test preprocessing
    print("🔍 Testing preprocessing...")
    X, y, scaler, encoder = load_and_preprocess("data/sample_energy_data.csv")
    print(f"✅ X shape: {X.shape}")
    print(f"✅ y shape: {y.shape}")
    print(f"✅ Month classes: {encoder.classes_}")
    print(f"✅ Feature range: [{X.min():.3f}, {X.max():.3f}]")


