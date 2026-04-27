from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib, numpy as np, pandas as pd

app = Flask(__name__)
CORS(app)

model     = joblib.load('saved_model/diabetes_model.pkl')
scaler    = joblib.load('saved_model/scaler.pkl')
threshold = joblib.load('saved_model/best_threshold.pkl')
features  = joblib.load('saved_model/feature_names.pkl')
NEEDS_SCALE = False

@app.route('/')
def home():
    return send_from_directory('.', 'diabetes_ui.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    missing = [f for f in features if f not in data]
    if missing:
        return jsonify({'error': f'Missing features: {missing}'}), 400
    X = pd.DataFrame([data], columns=features)
    X_input = scaler.transform(X) if NEEDS_SCALE else X
    prob = float(model.predict_proba(X_input)[0, 1])
    pred = int(prob >= threshold)
    return jsonify({
        'prediction': pred,
        'label': 'Diabetic' if pred == 1 else 'Not diabetic',
        'probability_diabetic': round(prob, 4),
        'threshold_used': round(float(threshold), 4),
        'confidence': round(prob if pred == 1 else 1 - prob, 4),
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'diabetes_classifier'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
