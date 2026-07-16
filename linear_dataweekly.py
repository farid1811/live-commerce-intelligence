import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# Load data
df = pd.read_excel('Data Skripsi full.xlsx', sheet_name='Sheet1', header=1)

print("REGRESI LINEAR DENGAN CONSTRAINTS - DATA WEEKLY")
print("="*60)

# DATA WEEKLY - Aggregasi per minggu
bulan_order = ['januari', 'februari', 'maret', 'april', 'mei', 'juni', 'juli', 'agustus', 'september', 'oktober', 'november', 'desember']
df['Bulan'] = pd.Categorical(df['Bulan'], categories=bulan_order, ordered=True)
df['Bulan_Str'] = df['Bulan'].astype(str)

df_sorted = df.sort_values('Bulan')
df_sorted['Week_Number'] = (df_sorted.groupby('Bulan_Str').cumcount() // 7) + 1
df_sorted['Month_Week'] = df_sorted['Bulan_Str'] + '_Week' + df_sorted['Week_Number'].astype(str)

df_weekly = df_sorted.groupby('Month_Week', observed=False).agg({
    'Durasi_Jam': 'mean',
    'Penonton Aktif': 'mean', 
    'Produk Terjual': 'mean',
    'Bulan_Str': 'first'
}).reset_index()

print(f"Data Weekly Statistics:")
print(f"Jumlah sampel: {len(df_weekly)}")
print(f"Durasi range: {df_weekly['Durasi_Jam'].min():.1f} - {df_weekly['Durasi_Jam'].max():.1f} jam")
print(f"Penonton range: {df_weekly['Penonton Aktif'].min():.0f} - {df_weekly['Penonton Aktif'].max():.0f} orang")
print(f"Produk range: {df_weekly['Produk Terjual'].min():.1f} - {df_weekly['Produk Terjual'].max():.1f} produk")

# Persiapan data weekly
X_weekly = df_weekly[['Durasi_Jam', 'Penonton Aktif']].values
y_weekly = df_weekly['Produk Terjual'].values

# Normalisasi
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_weekly)
y_scaled = scaler_y.fit_transform(y_weekly.reshape(-1, 1)).flatten()

# SGD dengan constraints: koefisien >= 0, intercept >= 0
def sgd_linear_constrained(X, y, learning_rate=0.01, epochs=1000):
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

# Training dengan constraints
print("\n" + "="*50)
print("TRAINING LINEAR REGRESSION DENGAN CONSTRAINTS")
print("="*50)

theta, bias, losses = sgd_linear_constrained(X_scaled, y_scaled, learning_rate=0.01, epochs=1000)

# Predictions
y_pred_scaled = np.dot(X_scaled, theta) + bias
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Evaluasi
mae = mean_absolute_error(y_weekly, y_pred)
mape = mean_absolute_percentage_error(y_weekly, y_pred)
rmse = np.sqrt(mean_squared_error(y_weekly, y_pred))

# Calculate R²
ss_res = np.sum((y_weekly - y_pred) ** 2)
ss_tot = np.sum((y_weekly - np.mean(y_weekly)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("\n" + "="*50)
print("HASIL REGRESI LINEAR - DATA WEEKLY")
print("="*50)

# Interpretasi koefisien asli
theta_original = theta / scaler_X.scale_
bias_original = bias - np.dot(scaler_X.mean_ / scaler_X.scale_, theta)
bias_original = scaler_y.inverse_transform([[bias_original]])[0][0]

print(f"Parameter model:")
print(f"Koef X1 (Durasi): {theta_original[0]:.6f}")
print(f"Koef X2 (Penonton): {theta_original[1]:.6f}")
print(f"Intercept: {bias_original:.6f}")

print(f"\nMetrik Evaluasi:")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"RMSE: {rmse:.4f}")

print(f"\nModel Linear (Dengan Constraints):")
print(f"Produk_Terjual = {theta_original[0]:.6f} × Durasi_Jam + {theta_original[1]:.6f} × Penonton_Aktif + {bias_original:.6f}")

# Test prediksi dengan constraints
print(f"\n" + "="*50)
print("TEST PREDIKSI DENGAN CONSTRAINTS")
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
    predicted = theta_original[0] * durasi + theta_original[1] * penonton + bias_original
    print(f"Durasi={durasi:2}j, Penonton={penonton:3} ({desc:15}) → {predicted:6.1f} produk")

# Validasi constraints
print(f"\n" + "="*50)
print("VALIDASI CONSTRAINTS")
print("="*50)

constraint1 = theta_original[0] >= 0
constraint2 = theta_original[1] >= 0
constraint3 = bias_original >= 0
constraint4 = r2 >= 0.4

print(f"✅ Koef X1 (Durasi) >= 0: {theta_original[0]:.6f} {'✓' if constraint1 else '✗'}")
print(f"✅ Koef X2 (Penonton) >= 0: {theta_original[1]:.6f} {'✓' if constraint2 else '✗'}")
print(f"✅ Intercept >= 0: {bias_original:.6f} {'✓' if constraint3 else '✗'}")
print(f"✅ R² >= 0.4: {r2:.4f} {'✓' if constraint4 else '✗'}")

all_constraints_met = all([constraint1, constraint2, constraint3, constraint4])
print(f"\nStatus: {'SEMUA SYARAT TERPENUHI ✅' if all_constraints_met else 'SOME CONSTRAINTS FAILED ❌'}")

# Visualisasi hasil
plt.figure(figsize=(15, 5))

# Plot 1: Actual vs Predicted
plt.subplot(1, 3, 1)
plt.scatter(y_weekly, y_pred, alpha=0.7, s=80, color='purple')
plt.plot([y_weekly.min(), y_weekly.max()], [y_weekly.min(), y_weekly.max()], 'r--', lw=2)
plt.xlabel('Actual Produk Terjual')
plt.ylabel('Predicted Produk Terjual')
plt.title('Actual vs Predicted\n(Data Weekly dengan Constraints)')
plt.grid(True, alpha=0.3)

# Plot 2: Loss convergence
plt.subplot(1, 3, 2)
plt.plot(range(0, 1000, 100), losses, marker='o', color='purple')
plt.xlabel('Epoch (x100)')
plt.ylabel('Loss (MSE)')
plt.title('Konvergensi SGD\n(Data Weekly)')
plt.grid(True, alpha=0.3)

# Plot 3: Trend mingguan
plt.subplot(1, 3, 3)
weeks = range(len(df_weekly))
plt.plot(weeks, y_weekly, marker='o', label='Actual', color='blue', alpha=0.7)
plt.plot(weeks, y_pred, marker='s', label='Predicted', color='purple', alpha=0.7)
plt.xlabel('Minggu ke-')
plt.ylabel('Produk Terjual')
plt.title('Trend Actual vs Predicted per Minggu')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Tampilkan sample prediksi
print(f"\n" + "="*50)
print("SAMPLE WEEKLY PREDIKSI (10 data pertama)")
print("="*50)

sample_results = pd.DataFrame({
    'Bulan': df_weekly['Bulan_Str'].head(10),
    'Week': df_weekly['Month_Week'].head(10),
    'Durasi_Jam': df_weekly['Durasi_Jam'].head(10),
    'Penonton_Aktif': df_weekly['Penonton Aktif'].head(10),
    'Actual': y_weekly[:10],
    'Predicted': y_pred[:10],
    'Error': np.abs(y_weekly[:10] - y_pred[:10])
})
print(sample_results.round(4))

# Summary statistics
print(f"\nSummary Statistics Data Weekly:")
print(f"Rata-rata Actual: {y_weekly.mean():.2f}")
print(f"Rata-rata Predicted: {y_pred.mean():.2f}")
print(f"Std Actual: {y_weekly.std():.2f}")
print(f"Std Predicted: {y_pred.std():.2f}")

# ==================== FINAL SUMMARY ====================

print(f"\n" + "="*80)
print("HASIL AKHIR SEMUA MODEL LINEAR DENGAN CONSTRAINTS")
print("="*80)
print(f"{'Data Type':<12} {'R²':<8} {'Koef X1':<10} {'Koef X2':<10} {'Intercept':<10} {'MAE':<8} {'Status':<10}")
print(f"{'-'*80}")

results_summary = [
    {'type': 'All Data', 'r2': 0.4999, 'coef1': 0.042597, 'coef2': 0.010654, 'intercept': 5.964056, 'mae': 10.0068, 'valid': True},
    {'type': 'Data Clean', 'r2': 0.5330, 'coef1': 0.006840, 'coef2': 0.018381, 'intercept': 8.627275, 'mae': 8.2306, 'valid': True},
    {'type': 'Data Monthly', 'r2': 0.5794, 'coef1': 0.241730, 'coef2': 0.031181, 'intercept': -21.639995, 'mae': 5.9470, 'valid': False},
    {'type': 'Data Weekly', 'r2': r2, 'coef1': theta_original[0], 'coef2': theta_original[1], 'intercept': bias_original, 'mae': mae, 'valid': all_constraints_met}
]

for result in results_summary:
    status = "VALID ✅" if result['valid'] else "INVALID ❌"
    print(f"{result['type']:<12} {result['r2']:<8.4f} {result['coef1']:<10.6f} "
          f"{result['coef2']:<10.6f} {result['intercept']:<10.6f} {result['mae']:<8.4f} {status}")

print(f"{'='*80}")

# Rekomendasi model terbaik
valid_models = [r for r in results_summary if r['valid']]
if valid_models:
    best_model = max(valid_models, key=lambda x: x['r2'])
    print(f"\n🏆 MODEL TERBAIK: {best_model['type']}")
    print(f"   - R²: {best_model['r2']:.4f}")
    print(f"   - MAE: {best_model['mae']:.4f}")
    print(f"   - Semua constraints terpenuhi")
else:
    print(f"\n❌ Tidak ada model yang memenuhi semua syarat constraints")