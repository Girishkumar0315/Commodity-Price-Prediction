"""
Commodity Price Prediction - ML Training, Testing & Statistical Analysis
Dataset: Indian Agricultural Market Prices
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

import os
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'static', 'charts')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. LOAD & PREPROCESS ────────────────────────────────────────────────────
print("=" * 60)
print("COMMODITY PRICE PREDICTION - ML PIPELINE")
print("=" * 60)

df = pd.read_csv('data/commodity_prices.csv')
df.columns = [c.replace('_x0020_', '_') for c in df.columns]

# Parse date
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], dayfirst=True, errors='coerce')
df['Month'] = df['Arrival_Date'].dt.month
df['Year']  = df['Arrival_Date'].dt.year
df['DayOfWeek'] = df['Arrival_Date'].dt.dayofweek
df.dropna(subset=['Month', 'Year'], inplace=True)

print(f"\n✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Commodities: {df['Commodity'].nunique()}")
print(f"   States     : {df['State'].nunique()}")
print(f"   Date range : {df['Arrival_Date'].min().date()} → {df['Arrival_Date'].max().date()}")

# ── 2. ENCODE CATEGORICALS ──────────────────────────────────────────────────
cat_cols = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

features = [c + '_enc' for c in cat_cols] + ['Min_Price', 'Max_Price', 'Month', 'Year', 'DayOfWeek']
target   = 'Modal_Price'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n📊 Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── 3. TRAIN MODELS ─────────────────────────────────────────────────────────
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression':  Ridge(alpha=1.0),
    'Random Forest':     RandomForestRegressor(n_estimators=30, max_depth=12, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
print("\n🤖 Training Models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X_train, y_train, cv=3, scoring='r2').mean()
    results[name] = {'model': model, 'predictions': y_pred,
                     'RMSE': rmse, 'MAE': mae, 'R2': r2, 'CV_R2': cv}
    print(f"   {name:<22} | R²={r2:.4f} | RMSE={rmse:,.0f} | MAE={mae:,.0f} | CV_R²={cv:.4f}")

# Save best model
best_name = max(results, key=lambda k: results[k]['R2'])
best_model = results[best_name]['model']
print(f"\n🏆 Best model: {best_name} (R²={results[best_name]['R2']:.4f})")

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump({'model': best_model, 'encoders': encoders,
                 'features': features, 'model_name': best_name}, f)
print("   Model saved → models/best_model.pkl")

# ── 4. STATISTICAL TESTS ────────────────────────────────────────────────────
print("\n📐 STATISTICAL TESTS")
print("-" * 50)

# T-Test: Wheat vs Onion modal prices
wheat = df[df['Commodity'] == 'Wheat']['Modal_Price']
onion = df[df['Commodity'] == 'Onion']['Modal_Price']
t_stat, t_p = stats.ttest_ind(wheat, onion)
print(f"\n🔬 T-Test (Wheat vs Onion Modal Price):")
print(f"   t={t_stat:.4f}, p={t_p:.4e} → {'Significant ✅' if t_p < 0.05 else 'Not Significant ❌'}")

# Z-Test: Min vs Max price
from scipy.stats import norm
n = len(df)
z_stat = (df['Min_Price'].mean() - df['Max_Price'].mean()) / \
         (np.sqrt(df['Min_Price'].std()**2/n + df['Max_Price'].std()**2/n))
z_p = 2 * (1 - norm.cdf(abs(z_stat)))
print(f"\n🔬 Z-Test (Min Price vs Max Price):")
print(f"   z={z_stat:.4f}, p={z_p:.4e} → {'Significant ✅' if z_p < 0.05 else 'Not Significant ❌'}")

# Hypothesis: Punjab vs Rajasthan modal price
punjab = df[df['State'] == 'Punjab']['Modal_Price']
rajasthan = df[df['State'] == 'Rajasthan']['Modal_Price']
h_stat, h_p = stats.ttest_ind(punjab, rajasthan)
print(f"\n🔬 Hypothesis Test (Punjab vs Rajasthan Modal Price):")
print(f"   H0: No difference in modal prices between states")
print(f"   t={h_stat:.4f}, p={h_p:.4e} → {'Reject H0 ✅' if h_p < 0.05 else 'Fail to Reject H0'}")

# Linear Regression summary
from sklearn.linear_model import LinearRegression as LR
lr = LR()
lr.fit(X_train[['Min_Price','Max_Price']], y_train)
corr_coef, corr_p = stats.pearsonr(df['Min_Price'], df['Modal_Price'])
print(f"\n🔬 Linear Regression (Min_Price → Modal_Price):")
print(f"   Pearson r={corr_coef:.4f}, p={corr_p:.4e}")
print(f"   Coefficient: {lr.coef_}")
print(f"   Intercept:   {lr.intercept_:.2f}")

# Shapiro-Wilk normality on a sample
sample = df['Modal_Price'].sample(200, random_state=42)
sw_stat, sw_p = stats.shapiro(sample)
print(f"\n🔬 Shapiro-Wilk Normality Test (Modal Price sample n=200):")
print(f"   W={sw_stat:.4f}, p={sw_p:.4e} → {'Not Normal ✅' if sw_p < 0.05 else 'Normal'}")

stat_results = {
    't_test':    {'stat': t_stat,  'p': t_p,  'label': 'T-Test (Wheat vs Onion)'},
    'z_test':    {'stat': z_stat,  'p': z_p,  'label': 'Z-Test (Min vs Max Price)'},
    'hyp_test':  {'stat': h_stat,  'p': h_p,  'label': 'Hypothesis (Punjab vs Rajasthan)'},
    'pearson':   {'stat': corr_coef,'p': corr_p,'label': 'Pearson Correlation'},
    'shapiro':   {'stat': sw_stat, 'p': sw_p,  'label': 'Shapiro-Wilk Normality'},
}

# ── 5. CHARTS ───────────────────────────────────────────────────────────────
print("\n📈 Generating Charts...")
plt.style.use('seaborn-v0_8-whitegrid')
model_order = ['Ridge Regression', 'Gradient Boosting', 'Random Forest', 'Linear Regression']
model_colors = ['#4a80ce', '#f29e4c', '#2cb67d', '#eb5e46']

# Chart 1: Model Comparison (3 subplots: MAE, RMSE, R² Score)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

mae_vals = [results[m]['MAE'] for m in model_order]
rmse_vals = [results[m]['RMSE'] for m in model_order]
r2_vals = [results[m]['R2'] for m in model_order]

# MAE Subplot
ax1.bar(model_order, mae_vals, color=model_colors, width=0.6)
ax1.set_title('MAE Comparison (lower = better)', fontsize=11, fontweight='bold')
ax1.set_ylabel('MAE (₹)')
ax1.set_xticklabels(model_order, rotation=20, ha='right', fontsize=9)
ax1.set_ylim(0, max(mae_vals) * 1.15)

# RMSE Subplot
ax2.bar(model_order, rmse_vals, color=model_colors, width=0.6)
ax2.set_title('RMSE Comparison (lower = better)', fontsize=11, fontweight='bold')
ax2.set_ylabel('RMSE (₹)')
ax2.set_xticklabels(model_order, rotation=20, ha='right', fontsize=9)
ax2.set_ylim(0, max(rmse_vals) * 1.15)

# R² Subplot
ax3.bar(model_order, r2_vals, color=model_colors, width=0.6)
ax3.set_title('R² Score (higher = better)', fontsize=11, fontweight='bold')
ax3.set_ylabel('R² Score')
ax3.set_xticklabels(model_order, rotation=20, ha='right', fontsize=9)
ax3.set_ylim(0, 1.15)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ model_comparison.png")

# Chart 2: Actual vs Predicted (best model)
best_pred = results[best_name]['predictions']
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, best_pred, alpha=0.5, color='#82adcf', edgecolors='none', s=25, label='_nolegend_')
lims = [0, max(y_test.max(), best_pred.max()) * 1.02]
ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Modal Price (₹)', fontsize=11)
ax.set_ylabel('Predicted Modal Price (₹)', fontsize=11)
ax.set_title(f'Actual vs Predicted — {best_name}', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', frameon=False)
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ actual_vs_predicted.png")

# Chart 3: Heatmap — Full Feature Correlation Matrix
fig, ax = plt.subplots(figsize=(10, 8))
corr_df = df[['State_enc','District_enc','Market_enc','Commodity_enc','Variety_enc','Grade_enc','Min_Price','Max_Price','Modal_Price']].copy()
corr_df.columns = ['State','District','Market','Commodity','Variety','Grade','Min Price','Max Price','Modal Price']
corr = corr_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1.0, vmax=1.0,
            linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 10, 'weight': 'bold'})
ax.set_title('Feature Correlation Matrix Heatmap (-1.0 to +1.0)', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("   ✅ heatmap.png")

# Chart 4: Top Commodities by Avg Modal Price
top_comm = df.groupby('Commodity')['Modal_Price'].mean().nlargest(12).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(top_comm.index, top_comm.values, color=plt.cm.viridis(np.linspace(0.2,0.9,12)))
for bar, val in zip(bars, top_comm.values):
    ax.text(val+50, bar.get_y()+bar.get_height()/2,
            f'₹{val:,.0f}', va='center', fontsize=9)
ax.set_title('Top 12 Commodities by Avg Modal Price', fontsize=13, fontweight='bold')
ax.set_xlabel('Avg Modal Price (₹)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/top_commodities.png', dpi=200, bbox_inches='tight')
plt.close()
print("   ✅ top_commodities.png")

# Chart 5: Price Distribution (violin)
top5 = df['Commodity'].value_counts().nlargest(5).index.tolist()
fig, ax = plt.subplots(figsize=(11, 6))
data_list = [df[df['Commodity']==c]['Modal_Price'].values for c in top5]
vp = ax.violinplot(data_list, showmeans=True, showmedians=True)
for pc, col in zip(vp['bodies'], model_colors[:4] + ['#7209b7']):
    pc.set_facecolor(col); pc.set_alpha(0.7)
ax.set_xticks(range(1, len(top5)+1)); ax.set_xticklabels(top5, rotation=15)
ax.set_title('Price Distribution — Top 5 Commodities (Violin Plot)', fontsize=13, fontweight='bold')
ax.set_ylabel('Modal Price (₹)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/price_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("   ✅ price_distribution.png")

# Chart 6: Monthly Avg Price Trend
monthly = df.groupby('Month')['Modal_Price'].mean()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(monthly.index, monthly.values, marker='o', color='#4361ee', linewidth=2.5, markersize=8)
ax.fill_between(monthly.index, monthly.values, alpha=0.15, color='#4361ee')
ax.set_xticks(range(1,13))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_title('Monthly Avg Modal Price Trend', fontsize=13, fontweight='bold')
ax.set_ylabel('Avg Modal Price (₹)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/monthly_trend.png', dpi=200, bbox_inches='tight')
plt.close()
print("   ✅ monthly_trend.png")

# Chart 7: State x Commodity Price Heatmap (Normalized -1.0 to +1.0)
top_commodities = df['Commodity'].value_counts().nlargest(12).index
top_states = df['State'].value_counts().nlargest(15).index
filtered_df = df[df['Commodity'].isin(top_commodities) & df['State'].isin(top_states)]
state_comm_pivot = filtered_df.pivot_table(values='Modal_Price', index='State', columns='Commodity', aggfunc='mean')

p_min = state_comm_pivot.min().min()
p_max = state_comm_pivot.max().max()
state_comm_norm = (state_comm_pivot - p_min) / (p_max - p_min) * 2.0 - 1.0

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(state_comm_norm, cmap='YlOrRd', ax=ax, linewidths=0.5, annot=True, fmt='.2f',
            vmin=-1.0, vmax=1.0, cbar_kws={'label':'Normalized Price Index (-1.0 to +1.0)'}, annot_kws={'size': 8})
ax.set_title('State × Commodity Avg Price Heatmap (-1.0 to +1.0)', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Commodity', fontsize=11); ax.set_ylabel('State', fontsize=11)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/state_month_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("   ✅ state_month_heatmap.png")

# Chart 8: Feature Importance (Random Forest) - Matching Screenshot 4 (Purple Horizontal Bars)
rf = results['Random Forest']['model']
feat_names = ['State_enc', 'District_enc', 'Market_enc', 'Commodity_enc', 'Variety_enc', 'Grade_enc', 'Min_Price', 'Max_Price']
importance_all = rf.feature_importances_
feat_map = {'State_enc': 0, 'District_enc': 1, 'Market_enc': 2, 'Commodity_enc': 3, 'Variety_enc': 4, 'Grade_enc': 5, 'Min_Price': 6, 'Max_Price': 7}
scores = [importance_all[feat_map[f]] for f in feat_names]
feat_series = pd.Series(scores, index=feat_names).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(feat_series.index, feat_series.values, color='#8b5cf6', height=0.6)
ax.set_title('Feature Importance — Random Forest', fontsize=12, fontweight='bold')
ax.set_xlabel('Importance Score', fontsize=11)
ax.set_xlim(0, max(feat_series.values) * 1.1)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ feature_importance.png")

# Chart 9: Statistical Test Results
fig, ax = plt.subplots(figsize=(9, 5))
test_labels = [v['label'] for v in stat_results.values()]
p_values    = [-np.log10(max(v['p'], 1e-300)) for v in stat_results.values()]
bar_colors  = ['#06d6a0' if v['p'] < 0.05 else '#f72585' for v in stat_results.values()]
bars = ax.barh(test_labels, p_values, color=bar_colors)
ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=1.5, label='p=0.05 threshold')
ax.set_xlabel('-log₁₀(p-value)'); ax.set_title('Statistical Tests — Significance (-log₁₀ p)', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/stat_tests.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ stat_tests.png")

# Chart 10: Linear Regression fit
fig, ax = plt.subplots(figsize=(8, 5))
x_range = np.linspace(df['Min_Price'].min(), df['Min_Price'].quantile(0.95), 100).reshape(-1,1)
ax.scatter(df['Min_Price'], df['Modal_Price'], alpha=0.15, s=8, color='#4361ee')
ax.plot(x_range, lr.predict(np.column_stack([x_range, x_range*1.2])),
        color='#f72585', linewidth=2.5, label='Regression line')
ax.set_xlabel('Min Price (₹)'); ax.set_ylabel('Modal Price (₹)')
ax.set_title(f'Linear Regression — Min→Modal Price (r={corr_coef:.3f})', fontsize=13, fontweight='bold')
ax.set_xlim(0, df['Min_Price'].quantile(0.95)); ax.set_ylim(0, df['Modal_Price'].quantile(0.95))
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/linear_regression.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ linear_regression.png")

print("\n✅ All charts saved to static/charts/")
print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}")

# Save key stats to a JSON for the web app
import json
summary = {
    'dataset': {'rows': len(df), 'commodities': df['Commodity'].nunique(),
                'states': df['State'].nunique(), 'date_range': f"{df['Arrival_Date'].min().date()} → {df['Arrival_Date'].max().date()}"},
    'models': {name: {'R2': round(r['R2'],4), 'RMSE': round(r['RMSE'],2),
                       'MAE': round(r['MAE'],2), 'CV_R2': round(r['CV_R2'],4)}
               for name, r in results.items()},
    'best_model': best_name,
    'stat_tests': {k: {'stat': round(float(v['stat']),4), 'p': float(v['p']),
                        'label': v['label'], 'significant': bool(v['p'] < 0.05)}
                   for k, v in stat_results.items()},
    'commodities': df['Commodity'].unique().tolist(),
    'states': df['State'].unique().tolist(),
}
with open('models/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("📄 Summary saved → models/summary.json")
