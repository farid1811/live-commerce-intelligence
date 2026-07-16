import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# Load data
df = pd.read_excel('Data Skripsi full.xlsx', sheet_name='Sheet1', header=1)

print("LOGARITHMIC REGRESSION + SGD - DATA MONTHLY")
print("="*60)

# DATA MONTHLY
bulan_order = ['januari', 'februari', 'maret', 'april', 'mei', 'juni', 'juli', 'agustus', 'september', 'oktober', 'november', 'desember']
df['Bulan'] = pd.Categorical(df['Bulan'], categories=bulan_order, ordered=True)

df_monthly = df.groupby('Bulan', observed=False).agg({
    'Durasi_Jam': 'mean',
    'Penonton Aktif': 'mean', 
    'Produk Terjual': 'mean'
}).reset_index()

print("Data Monthly (Rata-rata per Bulan):")
print(df_monthly.round(4))

# Persiapan data monthly
X_monthly = df_monthly[['Durasi_Jam', 'Penonton Aktif']].values
y_monthly = df_monthly['Produk Terjual'].values

print(f"\nData Monthly Statistics:")
print(f"Jumlah sampel: {len(X_monthly)}")
print(f"Durasi range: {X_monthly[:, 0].min():.1f} - {X_monthly[:, 0].max():.1f} jam")
print(f"Penonton range: {X_monthly[:, 1].min():.0f} - {X_monthly[:, 1].max():.0f} orang")
print(f"Produk range: {y_monthly.min():.1f} - {y_monthly.max():.1f} produk")

# Apply logarithmic transformation to features
X_log = np.log(X_monthly + 1)  # +1 to avoid log(0)

print(f"\nShape sebelum log: {X_monthly.shape}")
print(f"Shape setelah log: {X_log.shape}")
print(f"Sample data asli: {X_monthly[:3]}")
print(f"Sample data log: {X_log[:3]}")

# Normalisasi
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_log_scaled = scaler_X.fit_transform(X_log)
y_scaled = scaler_y.fit_transform(y_monthly.reshape(-1, 1)).flatten()

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

# Ensure no negative predictions
y_pred_log = np.maximum(y_pred_log, 0)

# Evaluasi
mae_log = mean_absolute_error(y_monthly, y_pred_log)
# Custom MAPE calculation to handle zeros
mape_log = np.mean(np.abs((y_monthly - y_pred_log) / np.maximum(y_monthly, 1))) * 100
rmse_log = np.sqrt(mean_squared_error(y_monthly, y_pred_log))

# Calculate R²
ss_res = np.sum((y_monthly - y_pred_log) ** 2)
ss_tot = np.sum((y_monthly - np.mean(y_monthly)) ** 2)
r2_log = 1 - (ss_res / ss_tot)

print("\n" + "="*50)
print("HASIL LOGARITHMIC REGRESSION - DATA MONTHLY")
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
    predicted = max(predicted, 0)  # Ensure no negative predictions
    
    print(f"Durasi={durasi:2}j, Penonton={penonton:3} ({desc:15}) → {predicted:6.1f} produk")

# Perbandingan dengan Linear Regression Monthly
print(f"\n" + "="*60)
print("PERBANDINGAN LINEAR vs LOGARITHMIC REGRESSION (DATA MONTHLY)")
print("="*60)
print(f"{'Metric':<10} {'Linear':<12} {'Logarithmic':<12} {'Improvement':<12}")
print(f"{'-'*60}")
print(f"{'R²':<10} {0.5794:<12.4f} {r2_log:<12.4f} {r2_log - 0.5794:>+10.4f}")
print(f"{'MAE':<10} {5.9470:<12.4f} {mae_log:<12.4f} {5.9470 - mae_log:>+10.4f}")
print(f"{'RMSE':<10} {6.9658:<12.4f} {rmse_log:<12.4f} {6.9658 - rmse_log:>+10.4f}")

if r2_log > 0.5794 and mae_log < 5.9470:
    print(f"\n🎉 LOGARITHMIC LEBIH BAIK dari Linear Regression!")
elif r2_log < 0.5794 and mae_log > 5.9470:
    print(f"\n❌ LOGARITHMIC LEBIH BURUK dari Linear Regression!")
else:
    print(f"\n📊 LOGARITHMIC dan Linear memiliki trade-off yang berbeda")

# Visualisasi hasil
plt.figure(figsize=(15, 10))

# Plot 1: Actual vs Predicted
plt.subplot(2, 3, 1)
plt.scatter(y_monthly, y_pred_log, alpha=0.7, s=100, color='purple')
plt.plot([y_monthly.min(), y_monthly.max()], [y_monthly.min(), y_monthly.max()], 'k--', lw=1)
plt.xlabel('Actual Produk Terjual')
plt.ylabel('Predicted Produk Terjual')
plt.title('Actual vs Predicted\n(Logarithmic Regression - Data Monthly)')
plt.grid(True, alpha=0.3)

# Plot 2: Loss convergence
plt.subplot(2, 3, 2)
plt.plot(range(0, 1000, 100), losses_log, marker='o', color='purple')
plt.xlabel('Epoch (x100)')
plt.ylabel('Loss (MSE)')
plt.title('Konvergensi SGD\n(Logarithmic Regression)')
plt.grid(True, alpha=0.3)

# Plot 3: Residuals
plt.subplot(2, 3, 3)
residuals = y_monthly - y_pred_log
plt.scatter(y_pred_log, residuals, alpha=0.7, s=80, color='purple')
plt.axhline(y=0, color='k', linestyle='--', lw=1)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot\n(Logarithmic Regression)')
plt.grid(True, alpha=0.3)

# Plot 4: Monthly comparison
plt.subplot(2, 3, 4)
months = df_monthly['Bulan']
x_pos = np.arange(len(months))
width = 0.35

plt.bar(x_pos - width/2, y_monthly, width, label='Actual', alpha=0.7, color='blue')
plt.bar(x_pos + width/2, y_pred_log, width, label='Predicted', alpha=0.7, color='purple')
plt.xlabel('Bulan')
plt.ylabel('Produk Terjual')
plt.title('Perbandingan Actual vs Predicted per Bulan')
plt.xticks(x_pos, months, rotation=45)
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

# Tampilkan semua prediksi monthly
print(f"\n" + "="*50)
print("MONTHLY PREDIKSI LOGARITHMIC")
print("="*50)

monthly_results = pd.DataFrame({
    'Bulan': df_monthly['Bulan'],
    'Durasi_Jam': df_monthly['Durasi_Jam'],
    'Penonton_Aktif': df_monthly['Penonton Aktif'],
    'Actual': y_monthly,
    'Predicted': y_pred_log,
    'Error': np.abs(y_monthly - y_pred_log)
})
print(monthly_results.round(4))

print(f"\n📊 INTERPRETASI LOGARITHMIC MODEL (DATA MONTHLY):")
print(f"- R² {r2_log:.4f} menunjukkan model menjelaskan {r2_log*100:.1f}% variasi data")
print(f"- Koefisien log(Penonton+1): {theta_log_original[1]:.6f} vs log(Durasi+1): {theta_log_original[0]:.6f}")
print(f"- MAPE {mape_log:.4f}% menunjukkan akurasi prediksi")
if bias_log_original < 0:
    print(f"- ⚠️  Intercept negatif: model mungkin kurang baik untuk nilai input rendah")
else:
    print(f"- ✅ Intercept positif: model reasonable untuk semua range input")

# FINAL COMPARISON ALL LOGARITHMIC MODELS
print(f"\n" + "="*90)
print("HASIL AKHIR SEMUA MODEL LOGARITHMIC REGRESSION")
print("="*90)
print(f"{'Data Type':<12} {'R²':<8} {'MAE':<8} {'RMSE':<8} {'Intercept':<12} {'Status':<10} {'vs Linear':<12}")
print(f"{'-'*90}")

log_results = [
    {'data': 'Weekly', 'r2': 0.6390, 'mae': 5.5196, 'rmse': 7.3053, 'intercept': -52.450910, 'linear_r2': 0.5469, 'linear_mae': 6.0432},
    {'data': 'Clean', 'r2': 0.5120, 'mae': 8.1781, 'rmse': 10.6252, 'intercept': -46.403496, 'linear_r2': 0.5330, 'linear_mae': 8.2306},
    {'data': 'All Data', 'r2': 0.4556, 'mae': 10.5492, 'rmse': 14.6269, 'intercept': -17.976276, 'linear_r2': 0.4999, 'linear_mae': 10.0068},
    {'data': 'Monthly', 'r2': r2_log, 'mae': mae_log, 'rmse': rmse_log, 'intercept': bias_log_original, 'linear_r2': 0.5794, 'linear_mae': 5.9470}
]

for result in log_results:
    status_icon = "✅" if result['intercept'] >= 0 else "❌"
    vs_linear = "BETTER" if result['r2'] > result['linear_r2'] and result['mae'] < result['linear_mae'] else "WORSE" if result['r2'] < result['linear_r2'] and result['mae'] > result['linear_mae'] else "MIXED"
    
    print(f"{result['data']:<12} {result['r2']:<8.4f} {result['mae']:<8.4f} {result['rmse']:<8.4f} {result['intercept']:<12.6f} {status_icon:<10} {vs_linear:<12}")

print(f"{'='*90}")

print(f"\n📈 TREND LOGARITHMIC REGRESSION:")
print(f"1. Data Weekly: Performa TERBAIK (R²: 0.6390, MAE: 5.5196)")
print(f"2. Masalah utama: INTERCEPT NEGATIF di semua data type")
print(f"3. Pattern: Semakin banyak data → semakin buruk performa")
print(f"4. Rekomendasi: Gunakan hanya untuk Data Weekly dengan perbaikan intercept")

print(f"\n🏆 KESIMPULAN FINAL LOGARITHMIC:")
print(f"- Logarithmic Regression cocok untuk Data Weekly (improvement signifikan)")
print(f"- Tidak recommended untuk All Data dan Data Clean (lebih buruk dari Linear)")
print(f"- Masalah intercept negatif perlu diatasi dengan constraint yang lebih ketat")