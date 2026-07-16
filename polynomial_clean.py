import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# Load data
df = pd.read_excel('Data Skripsi full.xlsx', sheet_name='Sheet1', header=1)

print("POLYNOMIAL REGRESSION (Degree=2) + SGD - DATA CLEAN")
print("="*60)

# DATA CLEAN
df_clean = df[df['Produk Terjual'] > 0].copy()

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in ['Durasi_Jam', 'Penonton Aktif', 'Produk Terjual']:
    df_clean = remove_outliers_iqr(df_clean, col)

print(f"Data Clean: {len(df_clean)} samples")

# Persiapan data
X_clean = df_clean[['Durasi_Jam', 'Penonton Aktif']].values
y_clean = df_clean['Produk Terjual'].values

# Create polynomial features (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_clean)

print(f"Shape sebelum polynomial: {X_clean.shape}")
print(f"Shape setelah polynomial: {X_poly.shape}")
print(f"Feature names: {poly.get_feature_names_out(['Durasi_Jam', 'Penonton_Aktif'])}")

# Normalisasi
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_poly_scaled = scaler_X.fit_transform(X_poly)
y_scaled = scaler_y.fit_transform(y_clean.reshape(-1, 1)).flatten()

# SGD untuk Polynomial Regression dengan constraints
def sgd_polynomial_constrained(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    bias = 0.1
    losses = []
    
    for epoch in range(epochs):
        for i in range(m):
            y_pred = np.dot(X[i], theta) + bias
            error = y_pred - y[i]
            
            grad_theta = error * X[i]
            grad_bias = error
            
            theta_new = theta - learning_rate * grad_theta
            bias_new = bias - learning_rate * grad_bias
            
            theta_new = np.maximum(theta_new, 0)
            bias_new = max(bias_new, 0)
            
            theta = theta_new
            bias = bias_new
        
        if epoch % 100 == 0:
            y_pred_all = np.dot(X, theta) + bias
            loss = np.mean((y_pred_all - y) ** 2)
            losses.append(loss)
    
    return theta, bias, losses

# Training polynomial regression
print("\n" + "="*50)
print("TRAINING POLYNOMIAL REGRESSION DENGAN CONSTRAINTS")
print("="*50)

theta_poly, bias_poly, losses_poly = sgd_polynomial_constrained(
    X_poly_scaled, y_scaled, learning_rate=0.01, epochs=1000
)

# Predictions
y_pred_poly_scaled = np.dot(X_poly_scaled, theta_poly) + bias_poly
y_pred_poly = scaler_y.inverse_transform(y_pred_poly_scaled.reshape(-1, 1)).flatten()

# Evaluasi
mae_poly = mean_absolute_error(y_clean, y_pred_poly)
mape_poly = mean_absolute_percentage_error(y_clean, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_clean, y_pred_poly))

# Calculate R²
ss_res = np.sum((y_clean - y_pred_poly) ** 2)
ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
r2_poly = 1 - (ss_res / ss_tot)

print("\n" + "="*50)
print("HASIL POLYNOMIAL REGRESSION - DATA CLEAN")
print("="*50)

# Interpretasi koefisien asli
theta_poly_original = theta_poly / scaler_X.scale_
bias_poly_original = bias_poly - np.dot(scaler_X.mean_ / scaler_X.scale_, theta_poly)
bias_poly_original = scaler_y.inverse_transform([[bias_poly_original]])[0][0]

feature_names = poly.get_feature_names_out(['Durasi_Jam', 'Penonton_Aktif'])

print(f"Parameter model (Polynomial):")
for i, (coef, name) in enumerate(zip(theta_poly_original, feature_names)):
    print(f"  {name}: {coef:.6f}")
print(f"Intercept: {bias_poly_original:.6f}")

print(f"\nMetrik Evaluasi:")
print(f"R²: {r2_poly:.4f}")
print(f"MAE: {mae_poly:.4f}")
print(f"MAPE: {mape_poly:.4f}%")
print(f"RMSE: {rmse_poly:.4f}")

# Validasi constraints
print(f"\n" + "="*50)
print("VALIDASI CONSTRAINTS")
print("="*50)

all_coef_positive = all(theta_poly_original >= 0)
intercept_positive = bias_poly_original >= 0
r2_valid = r2_poly >= 0.4

print(f"✅ Semua koefisien >= 0: {all_coef_positive} {'✓' if all_coef_positive else '✗'}")
print(f"✅ Intercept >= 0: {bias_poly_original:.6f} {'✓' if intercept_positive else '✗'}")
print(f"✅ R² >= 0.4: {r2_poly:.4f} {'✓' if r2_valid else '✗'}")

all_constraints_met = all([all_coef_positive, intercept_positive, r2_valid])
print(f"\nStatus: {'SEMUA SYARAT TERPENUHI ✅' if all_constraints_met else 'SOME CONSTRAINTS FAILED ❌'}")

# Test prediksi
print(f"\n" + "="*50)
print("TEST PREDIKSI POLYNOMIAL")
print("="*50)

test_cases = [
    (5, 10, "sangat rendah"),
    (8, 30, "rendah"),  
    (10, 50, "sedang"),
    (12, 80, "tinggi"),
    (15, 120, "sangat tinggi")
]

print("Prediksi untuk berbagai skenario:")
for durasi, penonton, desc in test_cases:
    X_test = np.array([[durasi, penonton]])
    X_test_poly = poly.transform(X_test)
    X_test_poly_scaled = scaler_X.transform(X_test_poly)
    
    predicted_scaled = np.dot(X_test_poly_scaled, theta_poly) + bias_poly
    predicted = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1))[0][0]
    
    print(f"Durasi={durasi:2}j, Penonton={penonton:3} ({desc:15}) → {predicted:6.1f} produk")

# Perbandingan dengan Linear Regression Data Clean
print(f"\n" + "="*60)
print("PERBANDINGAN LINEAR vs POLYNOMIAL REGRESSION (DATA CLEAN)")
print("="*60)
print(f"{'Metric':<10} {'Linear':<12} {'Polynomial':<12} {'Improvement':<12}")
print(f"{'-'*60}")
print(f"{'R²':<10} {0.5330:<12.4f} {r2_poly:<12.4f} {r2_poly - 0.5330:>+10.4f}")
print(f"{'MAE':<10} {8.2306:<12.4f} {mae_poly:<12.4f} {8.2306 - mae_poly:>+10.4f}")
print(f"{'RMSE':<10} {10.3931:<12.4f} {rmse_poly:<12.4f} {10.3931 - rmse_poly:>+10.4f}")

if r2_poly > 0.5330 and mae_poly < 8.2306:
    print(f"\n🎉 POLYNOMIAL LEBIH BAIK dari Linear Regression!")
elif r2_poly < 0.5330 and mae_poly > 8.2306:
    print(f"\n❌ POLYNOMIAL LEBIH BURUK dari Linear Regression!")
else:
    print(f"\n📊 POLYNOMIAL dan Linear memiliki trade-off yang berbeda")

# ==================== FINAL COMPARISON ====================

print(f"\n" + "="*80)
print("HASIL AKHIR SEMUA MODEL")
print("="*80)
print(f"{'Model':<20} {'Data Type':<12} {'R²':<8} {'MAE':<8} {'RMSE':<8} {'Status':<10}")
print(f"{'-'*80}")

final_results = [
    # Linear Models
    {'model': 'Linear', 'data': 'All Data', 'r2': 0.4999, 'mae': 10.0068, 'rmse': 14.0197, 'status': 'VALID'},
    {'model': 'Linear', 'data': 'Data Clean', 'r2': 0.5330, 'mae': 8.2306, 'rmse': 10.3931, 'status': 'VALID'},
    {'model': 'Linear', 'data': 'Data Weekly', 'r2': 0.5469, 'mae': 6.0432, 'rmse': 8.1847, 'status': 'VALID'},
    {'model': 'Linear', 'data': 'Data Monthly', 'r2': 0.5794, 'mae': 5.9470, 'rmse': 6.9658, 'status': 'INVALID'},
    
    # Polynomial Models
    {'model': 'Polynomial', 'data': 'Data Weekly', 'r2': 0.5339, 'mae': 6.1027, 'rmse': 8.3010, 'status': 'VALID'},
    {'model': 'Polynomial', 'data': 'Data Clean', 'r2': r2_poly, 'mae': mae_poly, 'rmse': rmse_poly, 'status': 'VALID' if all_constraints_met else 'INVALID'}
]

for result in final_results:
    status_icon = "✅" if result['status'] == 'VALID' else "❌"
    print(f"{result['model']:<20} {result['data']:<12} {result['r2']:<8.4f} {result['mae']:<8.4f} {result['rmse']:<8.4f} {status_icon}")

print(f"{'='*80}")

# Rekomendasi model terbaik
valid_models = [r for r in final_results if r['status'] == 'VALID']
if valid_models:
    best_model = min(valid_models, key=lambda x: x['mae'])
    print(f"\n🏆 MODEL TERBAIK: {best_model['model']} - {best_model['data']}")
    print(f"   - R²: {best_model['r2']:.4f}")
    print(f"   - MAE: {best_model['mae']:.4f}")
    print(f"   - RMSE: {best_model['rmse']:.4f}")
else:
    print(f"\n❌ Tidak ada model yang valid")

print(f"\n📊 KESIMPULAN:")
print(f"- Linear Regression lebih baik dari Polynomial untuk dataset ini")
print(f"- Data Weekly memberikan performa terbaik")
print(f"- Polynomial tidak memberikan improvement signifikan")