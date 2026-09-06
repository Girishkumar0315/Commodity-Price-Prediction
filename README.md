# 🌾 Commodity Price Prediction — ML Web Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0.0-000000?style=for-the-badge&logo=flask)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E?style=for-the-badge&logo=scikit-learn)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952B3?style=for-the-badge&logo=bootstrap)
![Dark Theme](https://img.shields.io/badge/Theme-Dark%20Mode-161b26?style=for-the-badge)

An advanced Machine Learning Web Application for predicting, analyzing, and comparing agricultural commodity market prices in India. Built with **Python, Flask, Scikit-Learn, Matplotlib, Seaborn, and Chart.js**.

---

## 🌐 Live Deployment

> [!IMPORTANT]
> **Live Demo Link**: Replace `YOUR_DEPLOYMENT_URL_HERE` below with your live website link once deployed (e.g. Render / Railway / Vercel):
>
> 🚀 **[Click Here to Open Live Application](https://commodityml.vercel.app)**  
> `URL Placeholder: https://commodity-price-prediction.onrender.com`

---

## 🌟 Key Features

* **🤖 Machine Learning Prediction Engine**: Predicts modal prices per quintal (100 kg) using trained **Random Forest** (R² = **99.76%**), **Gradient Boosting** (99.71%), **Linear Regression**, and **Ridge Regression** models.
* **📈 Dynamic Multi-Commodity Line Chart Comparison**: Every commodity predicted by the user automatically accumulates on an interactive Chart.js line chart for side-by-side comparison.
* **🔥 High-Resolution Correlation Matrix & Heatmaps**: Features a normalized **Feature Correlation Heatmap (-1.0 to +1.0)** and a **State × Commodity Average Price Heatmap**.
* **🔬 Rigorous Statistical Analysis**: Includes T-Test (Wheat vs Onion), Z-Test (Min vs Max Price), State Hypothesis Testing (Punjab vs Rajasthan), Pearson Correlation, and Shapiro-Wilk Normality tests.
* **🌙 Modern Dark-Themed UI**: Sleek, responsive dark mode design with modern cards, pill badges, and clean typography.

---

## 📊 Model Performance Benchmarks

| Model Architecture | MAE (₹) | RMSE (₹) | R² Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** ★ Best | **₹85.87** | **₹215.42** | **99.76%** | `Deployed` |
| **Gradient Boosting** | ₹108.91 | ₹234.93 | 99.71% | `Evaluated` |
| **Linear Regression** | ₹119.15 | ₹249.96 | 99.67% | `Evaluated` |
| **Ridge Regression** | ₹119.15 | ₹249.96 | 99.67% | `Evaluated` |

---

## 📁 Repository Directory Structure

```text
Commodity-Price-Prediction/
├── app.py                  # Flask Web Server & API Endpoints (/predict, /api/compare, etc.)
├── train_model.py          # ML Model Training Pipeline & High-Res Chart Generator
├── run.bat                 # One-click Windows Launcher Batch Script
├── run.sh                  # Linux/macOS Startup Bash Script
├── requirements.txt        # Python Dependencies
├── .gitignore              # Ignored files (pycache, env, build artifacts)
├── data/
│   └── commodity_prices.csv # Indian Agricultural Market Dataset
├── models/
│   ├── best_model.pkl      # Saved Scikit-Learn Model & LabelEncoders
│   └── summary.json        # Training Metrics & Dataset Summary JSON
├── static/
│   └── charts/             # Generated Matplotlib/Seaborn Chart Assets
└── templates/
    ├── base.html           # Dark Theme Parent Master Template & Navbar
    ├── index.html          # Overview Dashboard & Performance Summary
    ├── predict.html        # Prediction Engine & Dynamic Comparison Line Chart
    ├── charts.html         # Data Visualizations & Heatmap Analytics
    └── stats.html          # Inferential Statistical Hypotheses & Tests
```

---

## 💻 Local Installation & Setup

### Prerequisites
- Python **3.10+**
- `pip` package manager

### 1. Clone Repository
```bash
git clone https://github.com/Girishkumar0315/Commodity-Price-Prediction.git
cd Commodity-Price-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Models & Generate Charts (Optional)
```bash
python train_model.py
```

### 4. Run Flask Server
```bash
python app.py
```
Or on Windows, simply double-click **`run.bat`**.

Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 🚀 Deployment Instructions

To deploy this project to platforms like **Render**, **Railway**, **Vercel**, or **Heroku**:

1. Fork or push this repository to your GitHub account.
2. Create a new **Web Service** on Render (or your preferred cloud host).
3. Set the build & start commands:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app` (Linux/Cloud) or `python app.py`
4. Copy your live deployment URL and paste it in the **Live Deployment** section at the top of this `README.md`!

---

## 📜 License & Acknowledgements

* **Dataset**: Indian Agricultural Market Arrival & Price Data.
* **Author**: [Girishkumar0315](https://github.com/Girishkumar0315)
* Built for intelligent agricultural price prediction and market analytics.
