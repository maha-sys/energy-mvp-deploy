from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import io
from io import BytesIO
# ensure project root is on sys.path so sibling packages (like model/) can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# hi i am maha 




import tensorflow as tf
from optimization_engine import EnergyOptimizer
from model.predict_usage import predict_with_cost

app = Flask(__name__)
CORS(app)

# Paths


MODEL_PATH = '../model/energy_predictor.h5'

UPLOAD_FOLDER = '../uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize optimizer
optimizer = EnergyOptimizer()

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    model = None
    print("Model not found or failed to load:", e)

# Added now
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Energy Optimization API is running",
        "endpoints": {
            "health": "/health",
            "upload_csv": "/upload",
            "predict": "/predict",
            "optimize": "/optimize",
            "analytics": "/analytics",
            "recommendations": "/recommendations"
        }
    })
# upto this

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })


@app.route('/upload', methods=['GET','POST'])
def upload_file():
    # Provide a small browser form for manual testing
    if request.method == 'GET':
        return """
        <html><body>
        <h3>Upload CSV</h3>
        <form method="post" enctype="multipart/form-data">
          <input type="file" name="file" accept=".csv" />
          <button type="submit">Upload</button>
        </form>
        <p>Use this form to upload a CSV for testing the API.</p>
        </body></html>
        """

    # POST: actual upload handler
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files allowed'}), 400

    try:
        filepath = os.path.join(UPLOAD_FOLDER, 'user_energy_data.csv')
        file.save(filepath)

        df = pd.read_csv(filepath)

        # Normalize column names (show flexibility for user CSVs)
        def normalize_columns(df):
            col_map = {}
            for c in df.columns:
                key = c.strip().lower().replace(' ', '_').replace('-', '_')
                # month
                if 'month' in key:
                    col_map[c] = 'Month'
                # cost
                elif 'cost' in key or 'price' in key:
                    col_map[c] = 'Cost'
                # peak hours
                elif 'peak' in key:
                    col_map[c] = 'Peak_Usage_Hours'
                # average daily
                elif 'avg' in key or 'daily' in key:
                    col_map[c] = 'Avg_Daily_KWh'
                # units / kwh / usage
                elif 'unit' in key or 'kwh' in key or 'usage' in key:
                    # prefer avg vs total when indicated
                    if 'avg' in key or 'daily' in key:
                        col_map[c] = 'Avg_Daily_KWh'
                    else:
                        col_map[c] = 'Units_kWh'
            return df.rename(columns=col_map)

        df = normalize_columns(df)

        required_columns = [
            'Month',
            'Units_kWh',
            'Avg_Daily_KWh',
            'Peak_Usage_Hours',
            'Cost'
        ]

        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            return jsonify({
                'error': f'Missing columns: {missing}',
                'available_columns': list(df.columns)
            }), 400

        # Ensure numeric types
        for col in ['Units_kWh','Avg_Daily_KWh','Peak_Usage_Hours','Cost']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['Units_kWh','Avg_Daily_KWh','Peak_Usage_Hours'])

        # Save normalized CSV back to uploads so other endpoints use consistent data
        df.to_csv(filepath, index=False)

        # Helper to convert numbers to JSON-safe Python types (no NaN)
        def safe_number(x):
            try:
                if pd.isna(x):
                    return None
                return float(x)
            except Exception:
                return None

        stats = {
            'total_records': int(len(df)),
            'avg_units': safe_number(df['Units_kWh'].mean()),
            'total_units': safe_number(df['Units_kWh'].sum()),
            'avg_cost': safe_number(df['Cost'].dropna().mean())
        }

        return jsonify({
            'message': 'File uploaded successfully',
            'statistics': stats
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def fallback_predict_kwh(avg_daily, peak_hours):
    """Simple heuristic fallback used when TF/model is unavailable."""
    base = avg_daily * 30.0
    # Add up to +15% adjustment when peak hours are high
    factor = 1.0 + min(max(peak_hours, 0) / 24.0, 1.0) * 0.15
    return round(base * factor, 2)


@app.route('/predict', methods=['GET','POST'])
def predict_usage():
    # Small browser form for manual testing
    if request.method == 'GET':
        return """
        <html><body>
        <h3>Predict Usage</h3>
        <p>Submit values to get a predicted monthly kWh</p>
        <form id="predictForm">
          Month: <input name="Month" value="Jan"/><br/>
          Avg_Daily_KWh: <input name="Avg_Daily_KWh" value="10.5"/><br/>
          Peak_Usage_Hours: <input name="Peak_Usage_Hours" value="8"/><br/>
          Cost per kWh: <input name="cost_per_kwh" value="6.5"/><br/>
          <button type="button" onclick="submitPredict()">Predict</button>
        </form>
        <pre id="out"></pre>
        <script>
          async function submitPredict(){
            const f = document.getElementById('predictForm');
            const data = { Month: f.Month.value, Avg_Daily_KWh: f.Avg_Daily_KWh.value, Peak_Usage_Hours: f.Peak_Usage_Hours.value, cost_per_kwh: f.cost_per_kwh.value };
            const res = await fetch('/predict', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(data) });
            document.getElementById('out').textContent = await res.text();
          }
        </script>
        </body></html>
        """

    try:
        data = request.json or {}
        # Also accept form-encoded fallback from browser forms
        if not data and request.form:
            data = {k: request.form.get(k) for k in ['Month','Avg_Daily_KWh','Peak_Usage_Hours','cost_per_kwh']}

        required = ['Month', 'Avg_Daily_KWh', 'Peak_Usage_Hours']
        missing = [k for k in required if k not in data or data.get(k) in (None,'')]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        try:
            month = data['Month']
            avg_daily = float(data['Avg_Daily_KWh'])
            peak_hours = float(data['Peak_Usage_Hours'])
            cost_per_kwh = float(data.get('cost_per_kwh', 6.5))
        except Exception as e:
            return jsonify({'error': f'Invalid input types: {str(e)}'}), 400

        # If ML model is not loaded, use heuristic fallback
        if model is None:
            predicted_kwh = fallback_predict_kwh(avg_daily, peak_hours)
            estimated_cost = round(predicted_kwh * cost_per_kwh, 2)
            return jsonify({
                'predicted_units_kwh': predicted_kwh,
                'estimated_cost': estimated_cost,
                'cost_per_kwh': cost_per_kwh,
                'used_fallback': True,
                'timestamp': datetime.now().isoformat()
            })

        # Otherwise use the trained model helper
        try:
            result = predict_with_cost(month, avg_daily, peak_hours, cost_per_kwh=cost_per_kwh)
            return jsonify({
                'predicted_units_kwh': result['predicted_kwh'],
                'estimated_cost': result['estimated_cost'],
                'cost_per_kwh': result['cost_per_kwh'],
                'used_fallback': False,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/optimize', methods=['GET', 'POST'])
def optimize_energy():

    if request.method == 'GET':
        return """
        <html><body>
        <h3>Optimize Energy</h3>
        <form method="post">
          Target reduction (e.g., 0.15):
          <input name="target_reduction" value="0.15"/><br/>
          Time horizon (days):
          <input name="days" value="30"/><br/>
          <button type="submit">Run Optimize</button>
        </form>
        </body></html>
        """

    try:
        filepath = os.path.join(UPLOAD_FOLDER, 'user_energy_data.csv')
        if not os.path.exists(filepath):
            return jsonify({'error': 'Upload data first'}), 400

        df = pd.read_csv(filepath)

        # ✅ SAFE JSON handling
        data = request.get_json(silent=True)

        if not data and request.form:
            data = request.form.to_dict()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # ✅ ACCEPT BOTH frontend + form
        target_reduction = float(data.get('target_reduction', 0.15))
        days = int(float(data.get('days', data.get('time_horizon', 30))))

        recommendations = optimizer.optimize(
            df,
            target_reduction,
            days
        )

        return jsonify(recommendations)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analytics', methods=['GET'])
def get_analytics():
    try:
        filepath = os.path.join(UPLOAD_FOLDER, 'user_energy_data.csv')
        if not os.path.exists(filepath):
            return jsonify({'error': 'No data uploaded'}), 400

        df = pd.read_csv(filepath)

        # Convert groupby results to JSON-safe values
        def make_safe_dict(series):
            return {str(k): (None if pd.isna(v) else float(v)) for k, v in series.items()}

        analytics = {
            'monthly_avg_units': make_safe_dict(df.groupby('Month')['Units_kWh'].mean()),
            'peak_usage_impact': make_safe_dict(df.groupby('Peak_Usage_Hours')['Units_kWh'].mean()),
            'total_units': (None if pd.isna(df['Units_kWh'].sum()) else float(df['Units_kWh'].sum())),
            'total_cost': (None if pd.isna(df['Cost'].dropna().sum()) else float(df['Cost'].dropna().sum())),
            'trend': calculate_trend(df)
        }

        return jsonify(analytics)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def calculate_trend(df):
    if len(df) < 2:
        return 'insufficient_data'

    df = df.reset_index()
    corr = np.corrcoef(df.index, df['Units_kWh'])[0, 1]

    if corr > 0.1:
        return 'increasing'
    elif corr < -0.1:
        return 'decreasing'
    return 'stable'


@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    try:
        filepath = os.path.join(UPLOAD_FOLDER, 'user_energy_data.csv')
        if not os.path.exists(filepath):
            return jsonify({'error': 'No data uploaded'}), 400

        df = pd.read_csv(filepath)
        recommendations = optimizer.generate_recommendations(df)

        return jsonify(recommendations)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/reports/pdf', methods=['GET'])
def download_pdf_report():
    """Generate a simple PDF report server-side using reportlab when available.
    If reportlab is not installed, return a JSON error so the frontend can fallback.
    """
    try:
        filepath = os.path.join(UPLOAD_FOLDER, 'user_energy_data.csv')
        if not os.path.exists(filepath):
            return jsonify({'error': 'No data uploaded'}), 400

        df = pd.read_csv(filepath)

        # Prepare analytics
        def make_safe_dict(series):
            return {str(k): (None if pd.isna(v) else float(v)) for k, v in series.items()}

        analytics = {
            'monthly_avg_units': make_safe_dict(df.groupby('Month')['Units_kWh'].mean()),
            'total_units': (None if pd.isna(df['Units_kWh'].sum()) else float(df['Units_kWh'].sum())),
            'total_cost': (None if pd.isna(df['Cost'].dropna().sum()) else float(df['Cost'].dropna().sum())),
            'trend': calculate_trend(df)
        }

        recommendations = optimizer.generate_recommendations(df)

        # Try to import reportlab; if missing, tell frontend to fallback
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
        except Exception:
            return jsonify({'error': 'pdf_unavailable', 'message': 'reportlab is not installed on the server'}), 501

        # Build PDF in memory
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        y = 750
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "EnergyAI Optimization Report")
        y -= 30
        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Generated: {datetime.now().isoformat()}")
        y -= 20
        c.drawString(50, y, f"Total Units: {analytics.get('total_units')}")
        y -= 15
        c.drawString(50, y, f"Total Cost: {analytics.get('total_cost')}")
        y -= 25

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Monthly Averages:")
        y -= 18
        c.setFont("Helvetica", 10)
        for m, v in analytics.get('monthly_avg_units', {}).items():
            c.drawString(60, y, f"{m}: {v}")
            y -= 14
            if y < 50:
                c.showPage()
                y = 750

        y -= 10
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Recommendations:")
        y -= 18
        c.setFont("Helvetica", 10)

        recs = recommendations if isinstance(recommendations, list) else (recommendations.get('recommendations') if isinstance(recommendations, dict) else [])
        if not recs:
            c.drawString(60, y, "No recommendations available.")
            y -= 14
        else:
            for r in recs:
                title = r.get('title') or r.get('strategy') or 'Recommendation'
                desc = r.get('description') or r.get('explanation') or ''
                est_kwh = r.get('estimated_kwh_savings')
                est_inr = r.get('estimated_monthly_savings_inr')
                line = f"- {title} {('₹'+str(est_inr)) if est_inr else ''} {str(est_kwh)+' kWh' if est_kwh else ''}"
                c.drawString(60, y, line)
                y -= 14
                if desc:
                    c.drawString(70, y, desc[:160])
                    y -= 12
                if y < 50:
                    c.showPage()
                    y = 750

        c.save()
        buffer.seek(0)

        return send_file(buffer, as_attachment=True, download_name='energy_report.pdf', mimetype='application/pdf')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)