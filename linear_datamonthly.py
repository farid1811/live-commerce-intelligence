import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# Load data
df = pd.read_excel('Data Skripsi full.xlsx', sheet_name='Sheet1', header=1)

print("REGRESI LINEAR DENGAN CONSTRAINTS - DATA MONTHLY")
print("="*60)

# DATA MONTHLY - Aggregasi per bulan
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

# Normalisasi
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_monthly)
y_scaled = scaler_y.fit_transform(y_monthly.reshape(-1, 1)).flatten()

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
mae = mean_absolute_error(y_monthly, y_pred)
mape = mean_absolute_percentage_error(y_monthly, y_pred)
rmse = np.sqrt(mean_squared_error(y_monthly, y_pred))

# Calculate R²
ss_res = np.sum((y_monthly - y_pred) ** 2)
ss_tot = np.sum((y_monthly - np.mean(y_monthly)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("\n" + "="*50)
print("HASIL REGRESI LINEAR - DATA MONTHLY")
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
plt.scatter(y_monthly, y_pred, alpha=0.7, s=100, color='green')
plt.plot([y_monthly.min(), y_monthly.max()], [y_monthly.min(), y_monthly.max()], 'r--', lw=2)
plt.xlabel('Actual Produk Terjual')
plt.ylabel('Predicted Produk Terjual')
plt.title('Actual vs Predicted\n(Data Monthly dengan Constraints)')
plt.grid(True, alpha=0.3)

# Plot 2: Loss convergence
plt.subplot(1, 3, 2)
plt.plot(range(0, 1000, 100), losses, marker='o', color='green')
plt.xlabel('Epoch (x100)')
plt.ylabel('Loss (MSE)')
plt.title('Konvergensi SGD\n(Data Monthly)')
plt.grid(True, alpha=0.3)

# Plot 3: Prediksi per bulan
plt.subplot(1, 3, 3)
months = df_monthly['Bulan']
x_pos = np.arange(len(months))
width = 0.35
plt.bar(x_pos - width/2, y_monthly, width, label='Actual', alpha=0.7, color='blue')
plt.bar(x_pos + width/2, y_pred, width, label='Predicted', alpha=0.7, color='lightgreen')
plt.xlabel('Bulan')
plt.ylabel('Produk Terjual')
plt.title('Perbandingan Actual vs Predicted per Bulan')
plt.xticks(x_pos, months, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Tampilkan semua prediksi monthly
print(f"\n" + "="*50)
print("MONTHLY PREDIKSI")
print("="*50)

monthly_results = pd.DataFrame({
    'Bulan': df_monthly['Bulan'],
    'Durasi_Jam': df_monthly['Durasi_Jam'],
    'Penonton_Aktif': df_monthly['Penonton Aktif'],
    'Actual': y_monthly,
    'Predicted': y_pred,
    'Error': np.abs(y_monthly - y_pred)
})
print(monthly_results.round(4))

# Summary statistics
print(f"\nSummary Statistics Data Monthly:")
print(f"Rata-rata Actual: {y_monthly.mean():.2f}")
print(f"Rata-rata Predicted: {y_pred.mean():.2f}")
print(f"Std Actual: {y_monthly.std():.2f}")
print(f"Std Predicted: {y_pred.std():.2f}")

# Perbandingan dengan sebelumnya
print(f"\n" + "="*60)
print("PERBANDINGAN SEMUA MODEL")
print("="*60)
print(f"{'Metric':<10} {'All Data':<12} {'Data Clean':<12} {'Data Monthly':<12}")
print(f"{'-'*60}")
print(f"{'R²':<10} {0.4999:<12.4f} {0.5330:<12.4f} {r2:<12.4f}")
print(f"{'MAE':<10} {10.0068:<12.4f} {8.2306:<12.4f} {mae:<12.4f}")
print(f"{'RMSE':<10} {14.0197:<12.4f} {10.3931:<12.4f} {rmse:<12.4f}")
print(f"{'Samples':<10} {217:<12} {183:<12} {len(X_monthly):<12}")