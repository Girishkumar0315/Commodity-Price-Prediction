"""
Flask Web App — Commodity Price Prediction Dashboard
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle, json, os
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
BASE = os.path.dirname(__file__)

# Load model & summary
with open(os.path.join(BASE, 'models/best_model.pkl'), 'rb') as f:
    pkg = pickle.load(f)
model    = pkg['model']
encoders = pkg['encoders']
features = pkg['features']
model_name = pkg['model_name']

with open(os.path.join(BASE, 'models/summary.json')) as f:
    summary = json.load(f)

df_raw = pd.read_csv(os.path.join(BASE, 'data/commodity_prices.csv'))
df_raw.columns = [c.replace('_x0020_', '_') for c in df_raw.columns]

# ── ROUTES ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', summary=summary)

@app.route('/charts')
def charts():
    return render_template('charts.html', summary=summary)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    commodities = sorted(df_raw['Commodity'].unique().tolist())
    states      = sorted(df_raw['State'].unique().tolist())
    grades      = sorted(df_raw['Grade'].unique().tolist())
    varieties   = sorted(df_raw['Variety'].unique().tolist())

    prediction = None
    error      = None
    form_data  = {}

    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()
            row = {}
            for col in ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']:
                val = form_data.get(col, df_raw[col].mode()[0])
                le  = encoders[col]
                if val in le.classes_:
                    row[col + '_enc'] = le.transform([val])[0]
                else:
                    row[col + '_enc'] = 0

            min_p = form_data.get('Min_Price')
            max_p = form_data.get('Max_Price')
            month_v = form_data.get('Month')
            year_v = form_data.get('Year')
            dow_v = form_data.get('DayOfWeek')

            row['Min_Price']  = float(min_p) if min_p not in (None, '') else 1500.0
            row['Max_Price']  = float(max_p) if max_p not in (None, '') else 3000.0
            row['Month']      = int(month_v) if month_v not in (None, '') else 6
            row['Year']       = int(year_v) if year_v not in (None, '') else 2024
            row['DayOfWeek']  = int(dow_v) if dow_v not in (None, '') else 1

            X_in = pd.DataFrame([row])[features]
            pred = model.predict(X_in)[0]
            prediction = round(float(pred), 2)
        except Exception as e:
            error = str(e)

    districts = sorted(df_raw['District'].unique().tolist())
    markets   = sorted(df_raw['Market'].unique().tolist())

    return render_template('predict.html', commodities=commodities, states=states,
                           grades=grades, varieties=varieties, districts=districts,
                           markets=markets, prediction=prediction, error=error,
                           form_data=form_data, model_name=model_name, summary=summary)

@app.route('/stats')
def statistics():
    return render_template('stats.html', summary=summary)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json() or {}
    try:
        row = {}
        for col in ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']:
            val = data.get(col, df_raw[col].mode()[0])
            le  = encoders[col]
            row[col + '_enc'] = le.transform([val])[0] if val in le.classes_ else 0
        min_p = data.get('Min_Price')
        max_p = data.get('Max_Price')
        month_v = data.get('Month')
        year_v = data.get('Year')
        dow_v = data.get('DayOfWeek')

        row['Min_Price']  = float(min_p) if min_p not in (None, '') else 1500.0
        row['Max_Price']  = float(max_p) if max_p not in (None, '') else 3000.0
        row['Month']      = int(month_v) if month_v not in (None, '') else 6
        row['Year']       = int(year_v) if year_v not in (None, '') else 2024
        row['DayOfWeek']  = int(dow_v) if dow_v not in (None, '') else 1
        X_in = pd.DataFrame([row])[features]
        pred = float(model.predict(X_in)[0])
        return jsonify({'prediction': round(pred, 2), 'model': model_name, 'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

@app.route('/api/compare', methods=['POST'])
def api_compare():
    data = request.get_json() or {}
    selected_commodities = data.get('commodities', ['Wheat', 'Onion', 'Tomato', 'Potato', 'Apple'])
    if not isinstance(selected_commodities, list) or len(selected_commodities) == 0:
        selected_commodities = ['Wheat', 'Onion', 'Tomato', 'Potato', 'Apple']

    state = data.get('State', df_raw['State'].mode()[0])
    min_p = float(data.get('Min_Price', 1500))
    max_p = float(data.get('Max_Price', 3000))
    month_v = int(data.get('Month', 6))
    year_v = int(data.get('Year', 2024))
    dow_v = int(data.get('DayOfWeek', 1))

    comparison_results = []
    for comm in selected_commodities:
        sub_df = df_raw[df_raw['Commodity'] == comm]
        district_val = sub_df['District'].mode()[0] if not sub_df.empty else df_raw['District'].mode()[0]
        market_val   = sub_df['Market'].mode()[0] if not sub_df.empty else df_raw['Market'].mode()[0]
        variety_val  = sub_df['Variety'].mode()[0] if not sub_df.empty else df_raw['Variety'].mode()[0]
        grade_val    = sub_df['Grade'].mode()[0] if not sub_df.empty else df_raw['Grade'].mode()[0]

        row = {}
        defaults = {
            'State': state, 'District': district_val, 'Market': market_val,
            'Commodity': comm, 'Variety': variety_val, 'Grade': grade_val
        }
        for col in ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']:
            val = defaults[col]
            le = encoders[col]
            row[col + '_enc'] = le.transform([val])[0] if val in le.classes_ else 0

        row['Min_Price'] = min_p
        row['Max_Price'] = max_p
        row['Month']     = month_v
        row['Year']      = year_v
        row['DayOfWeek'] = dow_v

        X_in = pd.DataFrame([row])[features]
        pred = round(float(model.predict(X_in)[0]), 2)

        avg_hist_modal = round(float(sub_df['Modal_Price'].mean()), 2) if not sub_df.empty else pred
        avg_hist_min   = round(float(sub_df['Min_Price'].mean()), 2) if not sub_df.empty else min_p
        avg_hist_max   = round(float(sub_df['Max_Price'].mean()), 2) if not sub_df.empty else max_p

        comparison_results.append({
            'commodity': comm,
            'predicted_price': pred,
            'avg_modal_price': avg_hist_modal,
            'avg_min_price': avg_hist_min,
            'avg_max_price': avg_hist_max
        })

    return jsonify({
        'status': 'ok',
        'model': model_name,
        'comparison': comparison_results
    })

@app.route('/api/summary')
def api_summary():
    return jsonify(summary)

if __name__ == '__main__':
    import socket

    def get_available_port(preferred_port=5000):
        for p in range(preferred_port, preferred_port + 20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('0.0.0.0', p))
                    return p
                except OSError:
                    continue
        return preferred_port

    target_port = int(os.environ.get('PORT', 5000))
    port = get_available_port(target_port)

    print("\n" + "=" * 50)
    print("  AgriML Dashboard Server")
    print(f"  Local Host Link:   http://127.0.0.1:{port}")
    print(f"  Network Host Link: http://localhost:{port}")
    print("=" * 50 + "\n")

    app.run(host='0.0.0.0', port=port, debug=True)
