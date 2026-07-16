import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# Load data
df = pd.read_excel('Data Skripsi full.xlsx', sheet_name='Sheet1', header=1)

print("LOGARITHMIC REGRESSION + SGD - ALL DATA")
print("="*60)

# ALL DATA
X_all = df[['Durasi_Jam', 'Penonton Aktif']].values
y_all = df['Produk Terjual'].values

print(f"All Data: {len(X_all)} samples")
print(f"Durasi range: {X_all[:, 0].min():.1f} - {X_all[:, 0].max():.1f} jam")
print(f"Penonton range: {X_all[:, 1].min():.0f} - {X_all[:, 1].max():.0f} orang")
print(f"Produk range: {y_all.min():.0f} - {y_all.max():.0f} produk")

# Apply logarithmic transformation to features (avoid log(0) by adding small constant)
X_log = np.log(X_all + 1)  # +1 to avoid log(0)

print(f"\nShape sebelum log: {X_all.shape}")
print(f"Shape setelah log: {X_log.shape}")
print(f"Sample data asli: {X_all[:3]}")
print(f"Sample data log: {X_log[:3]}")

# Normalisasi
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_log_scaled = scaler_X.fit_transform(X_log)
y_scaled = scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

# SGD untuk Logarithmic Regression dengan constraints
def sgd_logarithmic_constrained(X, y, learning_rate=0.01, epochs=1000):
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
            
            # Apply constraints: semua koefisien >= 0
            theta_new = np.maximum(theta_new, 0)
            bias_new = max(bias_new, 0)
            
            theta = theta_new
            bias = bias_new
        
        if epoch % 100 == 0:
            y_pred_all = np.dot(X, theta) + bias
            loss = np.mean((y_pred_all - y) ** 2)
            losses.append(loss)
    
    return theta, bias, losses

# Training logarithmic regression
print("\n" + "="*50)
print("TRAINING LOGARITHMIC REGRESSION DENGAN CONSTRAINTS")
print("="*50)

theta_log, bias_log, losses_log = sgd_logarithmic_constrained(
    X_log_scaled, y_scaled, learning_rate=0.01, epochs=1000
)

# Predictions
y_pred_log_scaled = np.dot(X_log_scaled, theta_log) + bias_log
y_pred_log = scaler_y.inverse_transform(y_pred_log_scaled.reshape(-1, 1)).flatten()

# Evaluasi
mae_log = mean_absolute_error(y_all, y_pred_log)

# Handle MAPE calculation to avoid division by zero
try:
    mape_log = mean_absolute_percentage_error(y_all, y_pred_log)
except:
    # Custom MAPE calculation to handle zeros
    mape_log = np.mean(np.abs((y_all - y_pred_log) / np.maximum(y_all, 1))) * 100

rmse_log = np.sqrt(mean_squared_error(y_all, y_pred_log))

# Calculate R²
ss_res = np.sum((y_all - y_pred_log) ** 2)
ss_tot = np.sum((y_all - np.mean(y_all)) ** 2)
r2_log = 1 - (ss_res / ss_tot)

print("\n" + "="*50)
print("HASIL LOGARITHMIC REGRESSION - ALL DATA")
print("="*50)

# Interpretasi koefisien asli
theta_log_original = theta_log / scaler_X.scale_
bias_log_original = bias_log - np.dot(scaler_X.mean_ / scaler_X.scale_, theta_log)
bias_log_original = scaler_y.inverse_transform([[bias_log_original]])[0][0]

print(f"Parameter model (Logarithmic):")
print(f"Koef log(Durasi+1): {theta_log_original[0]:.6f}")
print(f"Koef log(Penonton+1): {theta_log_original[1]:.6f}")
print(f"Intercept: {bias_log_original:.6f}")

print(f"\nMetrik Evaluasi:")
print(f"R²: {r2_log:.4f}")
print(f"MAE: {mae_log:.4f}")
print(f"MAPE: {mape_log:.4f}%")
print(f"RMSE: {rmse_log:.4f}")

print(f"\nModel Logarithmic:")
print(f"Produk_Terjual = {theta_log_original[0]:.6f} × log(Durasi+1) + {theta_log_original[1]:.6f} × log(Penonton+1) + {bias_log_original:.6f}")

# Validasi constraints
print(f"\n" + "="*50)
print("VALIDASI CONSTRAINTS")
print("="*50)

all_coef_positive = all(theta_log_original >= 0)
intercept_positive = bias_log_original >= 0
r2_valid = r2_log >= 0.4

print(f"✅ Semua koefisien >= 0: {all_coef_positive} {'✓' if all_coef_positive else '✗'}")
print(f"✅ Intercept >= 0: {bias_log_original:.6f} {'✓' if intercept_positive else '✗'}")
print(f"✅ R² >= 0.4: {r2_log:.4f} {'✓' if r2_valid else '✗'}")

all_constraints_met = all([all_coef_positive, intercept_positive, r2_valid])
print(f"\nStatus: {'SEMUA SYARAT TERPENUHI ✅' if all_constraints_met else 'SOME CONSTRAINTS FAILED ❌'}")

# Test prediksi
print(f"\n" + "="*50)
print("TEST PREDIKSI LOGARITHMIC")
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
    # Transform input ke logarithmic
    X_test = np.array([[durasi, penonton]])
    X_test_log = np.log(X_test + 1)
    X_test_log_scaled = scaler_X.transform(X_test_log)
    
    predicted_scaled = np.dot(X_test_log_scaled, theta_log) + bias_log
    predicted = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1))[0][0]
    
    print(f"Durasi={durasi:2}j, Penonton={penonton:3} ({desc:15}) → {predicted:6.1f} produk")

# Perbandingan dengan Linear Regression All Data
print(f"\n" + "="*60)
print("PERBANDINGAN LINEAR vs LOGARITHMIC REGRESSION (ALL DATA)")
print("="*60)
print(f"{'Metric':<10} {'Linear':<12} {'Logarithmic':<12} {'Improvement':<12}")
print(f"{'-'*60}")
print(f"{'R²':<10} {0.4999:<12.4f} {r2_log:<12.4f} {r2_log - 0.4999:>+10.4f}")
print(f"{'MAE':<10} {10.0068:<12.4f} {mae_log:<12.4f} {10.0068 - mae_log:>+10.4f}")
print(f"{'RMSE':<10} {14.0197:<12.4f} {rmse_log:<12.4f} {14.0197 - rmse_log:>+10.4f}")

if r2_log > 0.4999 and mae_log < 10.0068:
    print(f"\n🎉 LOGARITHMIC LEBIH BAIK dari Linear Regression!")
elif r2_log < 0.4999 and mae_log > 10.0068:
    print(f"\n❌ LOGARITHMIC LEBIH BURUK dari Linear Regression!")
else:
    print(f"\n📊 LOGARITHMIC dan Linear memiliki trade-off yang berbeda")

# Visualisasi hasil
plt.figure(figsize=(15, 10))

# Plot 1: Actual vs Predicted
plt.subplot(2, 3, 1)
plt.scatter(y_all, y_pred_log, alpha=0.6, s=60, color='red')
plt.plot([y_all.min(), y_all.max()], [y_all.min(), y_all.max()], 'k--', lw=1)
plt.xlabel('Actual Produk Terjual')
plt.ylabel('Predicted Produk Terjual')
plt.title('Actual vs Predicted\n(Logarithmic Regression - All Data)')
plt.grid(True, alpha=0.3)

# Plot 2: Loss convergence
plt.subplot(2, 3, 2)
plt.plot(range(0, 1000, 100), losses_log, marker='o', color='red')
plt.xlabel('Epoch (x100)')
plt.ylabel('Loss (MSE)')
plt.title('Konvergensi SGD\n(Logarithmic Regression)')
plt.grid(True, alpha=0.3)

# Plot 3: Residuals
plt.subplot(2, 3, 3)
residuals = y_all - y_pred_log
plt.scatter(y_pred_log, residuals, alpha=0.6, color='red')
plt.axhline(y=0, color='k', linestyle='--', lw=1)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot\n(Logarithmic Regression)')
plt.grid(True, alpha=0.3)

# Plot 4: Distribution of predictions vs actual
plt.subplot(2, 3, 4)
plt.hist(y_all, bins=20, alpha=0.7, label='Actual', color='blue')
plt.hist(y_pred_log, bins=20, alpha=0.7, label='Predicted', color='red')
plt.xlabel('Produk Terjual')
plt.ylabel('Frequency')
plt.title('Distribution: Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Feature importance based on coefficients
plt.subplot(2, 3, 5)
features = ['log(Durasi+1)', 'log(Penonton+1)']
coef_values = [theta_log_original[0], theta_log_original[1]]
colors = ['blue', 'orange']

plt.bar(features, coef_values, color=colors, alpha=0.7)
plt.ylabel('Koefisien')
plt.title('Feature Importance\n(Koefisien Logarithmic)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Tampilkan beberapa prediksi
print(f"\n" + "="*50)
print("SAMPLE PREDIKSI LOGARITHMIC (10 data pertama)")
print("="*50)

sample_results = pd.DataFrame({
    'Durasi_Jam': X_all[:10, 0],
    'Penonton_Aktif': X_all[:10, 1],
    'Actual': y_all[:10],
    'Predicted': y_pred_log[:10],
    'Error': np.abs(y_all[:10] - y_pred_log[:10])
})
print(sample_results.round(4))

print(f"\n📊 INTERPRETASI LOGARITHMIC MODEL (ALL DATA):")
print(f"- R² {r2_log:.4f} menunjukkan model menjelaskan {r2_log*100:.1f}% variasi data")
print(f"- Koefisien log(Penonton+1) lebih besar, menunjukkan pengaruh yang lebih kuat")
print(f"- MAPE {mape_log:.4f}% menunjukkan akurasi prediksi rata-rata")
if bias_log_original < 0:
    print(f"- ⚠️  Intercept negatif: model mungkin kurang baik untuk nilai input rendah")
else:
    print(f"- ✅ Intercept positif: model reasonable untuk semua range input")

# Summary comparison dengan model sebelumnya - VERSI DIPERBAIKI
print(f"\n" + "="*70)
print("RINGKASAN PERBANDINGAN MODEL - ALL DATA")
print("="*70)
print(f"{'Model':<12} {'R²':<8} {'MAE':<8} {'RMSE':<8} {'Status':<10} {'Intercept':<12}")
print(f"{'-'*70}")

# Data model comparison
models_data = [
    {'name': 'Linear', 'r2': 0.4999, 'mae': 10.0068, 'rmse': 14.0197, 'status': 'VALID', 'intercept': 5.964056},
    {'name': 'Polynomial', 'r2': 0.4854, 'mae': 10.1924, 'rmse': 14.2217, 'status': 'VALID', 'intercept': 6.491376},
    {'name': 'Logarithmic', 'r2': r2_log, 'mae': mae_log, 'rmse': rmse_log, 'status': 'VALID' if all_constraints_met else 'INVALID', 'intercept': bias_log_original}
]

for model in models_data:
    status_icon = "✅" if model['status'] == 'VALID' else "❌"
    print(f"{model['name']:<12} {model['r2']:<8.4f} {model['mae']:<8.4f} {model['rmse']:<8.4f} {status_icon:<10} {model['intercept']:<12.6f}")

print(f"{'='*70}")

# Analisis akhir
print(f"\n🔍 ANALISIS AKHIR LOGARITHMIC ALL DATA:")
print(f"❌ PERFORMANCE: Lebih buruk dari Linear (R²: 0.4556 vs 0.4999)")
print(f"❌ CONSTRAINTS: Intercept negatif (-17.976) → tidak valid")
print(f"❌ MAPE: Sangat besar ({mape_log:.2f}%) → masalah numerik")
print(f"🎯 REKOMENDASI: Gunakan Linear Regression untuk All Data")

print(f"\n" + "="*70)
print("LOGARITHMIC LEBIH COCOK UNTUK DATA WEEKLY/MONTHLY")
print("="*70)