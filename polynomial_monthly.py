import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# Load data
df = pd.read_excel('Data Skripsi full.xlsx', sheet_name='Sheet1', header=1)

print("POLYNOMIAL REGRESSION (Degree=2) + SGD - DATA MONTHLY")
print("="*60)

# DATA MONTHLY
bulan_order = ['januari', 'februari', 'maret', 'april', 'mei', 'juni', 'juli', 'agustus', 'september', 'oktober', 'november', 'desember']
df['Bulan'] = pd.Categorical(df['Bulan'], categories=bulan_order, ordered=True)

df_monthly = df.groupby('Bulan', observed=False).agg({
    'Durasi_Jam': 'mean',
    'Penonton Aktif': 'mean', 
    'Produk Terjual': 'mean'
}).reset_index()

print("Data Monthly:")
print(df_monthly.round(4))

# Persiapan data
X_monthly = df_monthly[['Durasi_Jam', 'Penonton Aktif']].values
y_monthly = df_monthly['Produk Terjual'].values

print(f"\nData Monthly Statistics:")
print(f"Jumlah sampel: {len(X_monthly)}")
print(f"Durasi range: {X_monthly[:, 0].min():.1f} - {X_monthly[:, 0].max():.1f} jam")
print(f"Penonton range: {X_monthly[:, 1].min():.0f} - {X_monthly[:, 1].max():.0f} orang")
print(f"Produk range: {y_monthly.min():.1f} - {y_monthly.max():.1f} produk")

# Create polynomial features (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_monthly)

print(f"\nShape sebelum polynomial: {X_monthly.shape}")
print(f"Shape setelah polynomial: {X_poly.shape}")
print(f"Feature names: {poly.get_feature_names_out(['Durasi_Jam', 'Penonton_Aktif'])}")

# Normalisasi
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_poly_scaled = scaler_X.fit_transform(X_poly)
y_scaled = scaler_y.fit_transform(y_monthly.reshape(-1, 1)).flatten()

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
            
            # Apply constraints: semua koefisien >= 0, intercept >= 0
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
mae_poly = mean_absolute_error(y_monthly, y_pred_poly)
mape_poly = mean_absolute_percentage_error(y_monthly, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_monthly, y_pred_poly))

# Calculate R²
ss_res = np.sum((y_monthly - y_pred_poly) ** 2)
ss_tot = np.sum((y_monthly - np.mean(y_monthly)) ** 2)
r2_poly = 1 - (ss_res / ss_tot)

print("\n" + "="*50)
print("HASIL POLYNOMIAL REGRESSION - DATA MONTHLY")
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

# Perbandingan dengan Linear Regression Monthly
print(f"\n" + "="*60)
print("PERBANDINGAN LINEAR vs POLYNOMIAL REGRESSION (DATA MONTHLY)")
print("="*60)
print(f"{'Metric':<10} {'Linear':<12} {'Polynomial':<12} {'Improvement':<12}")
print(f"{'-'*60}")
print(f"{'R²':<10} {0.5794:<12.4f} {r2_poly:<12.4f} {r2_poly - 0.5794:>+10.4f}")
print(f"{'MAE':<10} {5.9470:<12.4f} {mae_poly:<12.4f} {5.9470 - mae_poly:>+10.4f}")
print(f"{'RMSE':<10} {6.9658:<12.4f} {rmse_poly:<12.4f} {6.9658 - rmse_poly:>+10.4f}")

if r2_poly > 0.5794 and mae_poly < 5.9470:
    print(f"\n🎉 POLYNOMIAL LEBIH BAIK dari Linear Regression!")
elif r2_poly < 0.5794 and mae_poly > 5.9470:
    print(f"\n❌ POLYNOMIAL LEBIH BURUK dari Linear Regression!")
else:
    print(f"\n📊 POLYNOMIAL dan Linear memiliki trade-off yang berbeda")

# ==================== FINAL COMPARISON ALL MODELS ====================

print(f"\n" + "="*90)
print("HASIL AKHIR SEMUA MODEL - LINEAR vs POLYNOMIAL (LENGKAP)")
print("="*90)
print(f"{'Model':<12} {'Data Type':<12} {'R²':<8} {'MAE':<8} {'RMSE':<8} {'Status':<10} {'Best':<8}")
print(f"{'-'*90}")

final_results = [
    # Linear Models
    {'model': 'Linear', 'data': 'All Data', 'r2': 0.4999, 'mae': 10.0068, 'rmse': 14.0197, 'status': 'VALID', 'best': ''},
    {'model': 'Linear', 'data': 'Data Clean', 'r2': 0.5330, 'mae': 8.2306, 'rmse': 10.3931, 'status': 'VALID', 'best': ''},
    {'model': 'Linear', 'data': 'Data Weekly', 'r2': 0.5469, 'mae': 6.0432, 'rmse': 8.1847, 'status': 'VALID', 'best': '🏆'},
    {'model': 'Linear', 'data': 'Data Monthly', 'r2': 0.5794, 'mae': 5.9470, 'rmse': 6.9658, 'status': 'INVALID', 'best': ''},
    
    # Polynomial Models
    {'model': 'Polynomial', 'data': 'All Data', 'r2': 0.4854, 'mae': 10.1924, 'rmse': 14.2217, 'status': 'VALID', 'best': ''},
    {'model': 'Polynomial', 'data': 'Data Clean', 'r2': 0.5246, 'mae': 8.3170, 'rmse': 10.4873, 'status': 'VALID', 'best': ''},
    {'model': 'Polynomial', 'data': 'Data Weekly', 'r2': 0.5339, 'mae': 6.1027, 'rmse': 8.3010, 'status': 'VALID', 'best': ''},
    {'model': 'Polynomial', 'data': 'Data Monthly', 'r2': r2_poly, 'mae': mae_poly, 'rmse': rmse_poly, 'status': 'VALID' if all_constraints_met else 'INVALID', 'best': ''},
]

for result in final_results:
    status_icon = "✅" if result['status'] == 'VALID' else "❌"
    best_icon = result['best']
    print(f"{result['model']:<12} {result['data']:<12} {result['r2']:<8.4f} {result['mae']:<8.4f} {result['rmse']:<8.4f} {status_icon:<10} {best_icon:<8}")

print(f"{'='*90}")

# Rekomendasi model terbaik
valid_models = [r for r in final_results if r['status'] == 'VALID']
if valid_models:
    best_overall = min(valid_models, key=lambda x: x['mae'])
    print(f"\n🏆 MODEL TERBAIK OVERALL: {best_overall['model']} - {best_overall['data']}")
    print(f"   - R²: {best_overall['r2']:.4f}")
    print(f"   - MAE: {best_overall['mae']:.4f}")
    print(f"   - RMSE: {best_overall['rmse']:.4f}")

# Visualisasi perbandingan semua model valid
plt.figure(figsize=(15, 10))

# Filter hanya model valid
valid_for_plot = [r for r in final_results if r['status'] == 'VALID']

# Plot 1: R² Comparison
plt.subplot(2, 3, 1)
models_plot = [f"{r['model']}\n{r['data']}" for r in valid_for_plot]
r2_values = [r['r2'] for r in valid_for_plot]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']

bars = plt.bar(models_plot, r2_values, color=colors, alpha=0.7)
plt.axhline(y=0.4, color='r', linestyle='--', label='Min R²=0.4')
plt.ylabel('R²')
plt.title('Perbandingan R² Semua Model Valid\n(Higher is Better)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: MAE Comparison
plt.subplot(2, 3, 2)
mae_values = [r['mae'] for r in valid_for_plot]
bars = plt.bar(models_plot, mae_values, color=colors, alpha=0.7)
plt.ylabel('MAE')
plt.title('Perbandingan MAE Semua Model Valid\n(Lower is Better)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot 3: RMSE Comparison
plt.subplot(2, 3, 3)
rmse_values = [r['rmse'] for r in valid_for_plot]
bars = plt.bar(models_plot, rmse_values, color=colors, alpha=0.7)
plt.ylabel('RMSE')
plt.title('Perbandingan RMSE Semua Model Valid\n(Lower is Better)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot 4: Linear vs Polynomial untuk setiap data type
plt.subplot(2, 3, 4)
data_types = ['All Data', 'Data Clean', 'Data Weekly', 'Data Monthly']
linear_r2 = [0.4999, 0.5330, 0.5469, 0.5794]
poly_r2 = [0.4854, 0.5246, 0.5339, r2_poly]

x_pos = np.arange(len(data_types))
width = 0.35

plt.bar(x_pos - width/2, linear_r2, width, label='Linear', alpha=0.7, color='blue')
plt.bar(x_pos + width/2, poly_r2, width, label='Polynomial', alpha=0.7, color='red')
plt.ylabel('R²')
plt.title('Linear vs Polynomial R² per Data Type')
plt.xticks(x_pos, data_types, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Linear vs Polynomial MAE
plt.subplot(2, 3, 5)
linear_mae = [10.0068, 8.2306, 6.0432, 5.9470]
poly_mae = [10.1924, 8.3170, 6.1027, mae_poly]

plt.bar(x_pos - width/2, linear_mae, width, label='Linear', alpha=0.7, color='blue')
plt.bar(x_pos + width/2, poly_mae, width, label='Polynomial', alpha=0.7, color='red')
plt.ylabel('MAE')
plt.title('Linear vs Polynomial MAE per Data Type')
plt.xticks(x_pos, data_types, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n📊 KESIMPULAN FINAL LENGKAP:")
print(f"1. Linear Regression lebih konsisten baik daripada Polynomial untuk semua data type")
print(f"2. Data Weekly memberikan performa terbaik untuk model valid")
print(f"3. Data Monthly memiliki performa terbaik secara metrik tapi INVALID (intercept negatif)")
print(f"4. Polynomial Regression tidak memberikan improvement signifikan")
print(f"5. Model terbaik: LINEAR REGRESSION dengan DATA WEEKLY")
print(f"6. Persamaan terbaik: Produk_Terjual = 0.073479 × Durasi + 0.020270 × Penonton + 2.742357")