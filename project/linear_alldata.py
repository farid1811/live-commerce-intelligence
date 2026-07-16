import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# =========================
# LOAD DATA
# =========================
df = pd.read_excel('Data Skripsi full.xlsx', sheet_name='Sheet1', header=1)

print("REGRESI LINEAR DENGAN CONSTRAINTS - ALL DATA")
print("="*60)

X = df[['Durasi_Jam', 'Penonton Aktif']].values
y = df['Produk Terjual'].values

print(f"Jumlah data: {len(X)}")
print(f"Jumlah data Produk Terjual = 0: {np.sum(y == 0)}")

# =========================
# NORMALISASI
# =========================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# =========================
# PARAMETER
# =========================
learning_rate = 0.01
max_epoch = 1000
tolerance = 0.001

m, n = X_scaled.shape
theta = np.zeros(n)
bias = 0.1

rmse_prev = float('inf')

print("\nTRAINING MODEL + KONVERGENSI")
print("="*60)
print("Iterasi | RMSE | Delta")

# =========================
# TRAINING
# =========================
for epoch in range(1, max_epoch+1):
    for i in range(m):
        y_pred = np.dot(X_scaled[i], theta) + bias
        error = y_pred - y_scaled[i]

        grad_theta = error * X_scaled[i]
        grad_bias = error

        theta = theta - learning_rate * grad_theta
        bias = bias - learning_rate * grad_bias

        # constraints
        theta = np.maximum(theta, 0)
        bias = max(bias, 0)

    y_pred_all = np.dot(X_scaled, theta) + bias
    rmse = np.sqrt(mean_squared_error(y_scaled, y_pred_all))
    delta = abs(rmse_prev - rmse)

    print(f"{epoch:6d} | {rmse:8.6f} | {delta:8.6f}")

    if delta < tolerance:
        print(f"\n✅ KONVERGEN di iterasi ke-{epoch}")
        break

    rmse_prev = rmse

else:
    print("\n⚠️ Berhenti karena maksimum iterasi")

# =========================
# KEMBALIKAN KE SKALA ASLI (PERSAMAAN TETAP SAMA)
# =========================
theta_original = theta / scaler_X.scale_
bias_original = bias - np.dot(scaler_X.mean_ / scaler_X.scale_, theta)
bias_original = scaler_y.inverse_transform([[bias_original]])[0][0]

# =========================
# PREDIKSI AKHIR
# =========================
y_pred_scaled = np.dot(X_scaled, theta) + bias
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# =========================
# EVALUASI
# =========================
mae = mean_absolute_error(y, y_pred)
rmse_real = np.sqrt(mean_squared_error(y, y_pred))

# R2
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

# =========================
# MAPE FIX (TANPA UBAH MODEL)
# =========================
mask = y != 0
mape = mean_absolute_percentage_error(y[mask], y_pred[mask]) * 100

# =========================
# OUTPUT HASIL
# =========================
print("\n" + "="*60)
print("HASIL REGRESI LINEAR - ALL DATA")
print("="*60)

print(f"Koef X1 (Durasi): {theta_original[0]:.6f}")
print(f"Koef X2 (Penonton): {theta_original[1]:.6f}")
print(f"Intercept: {bias_original:.6f}")

print("\nMetrik Evaluasi:")
print(f"R²   : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse_real:.4f}")
print(f"MAPE : {mape:.2f}%")

print("\nPersamaan:")
print(f"Y = {theta_original[0]:.6f}*X1 + {theta_original[1]:.6f}*X2 + {bias_original:.6f}")

# =========================
# SAMPLE DATA
# =========================
print("\nSAMPLE PREDIKSI (10 DATA PERTAMA)")
print("="*60)

sample = pd.DataFrame({
    "Durasi_Jam": X[:10, 0],
    "Penonton_Aktif": X[:10, 1],
    "Actual": y[:10],
    "Predicted": y_pred[:10]
})

sample["Error"] = sample["Actual"] - sample["Predicted"]

print(sample.round(2))

# =========================
# VISUALISASI
# =========================
print("\nMENAMPILKAN GRAFIK...")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(y, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")

plt.subplot(1,2,2)
residual = y - y_pred
plt.scatter(y_pred, residual, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual Plot")

plt.tight_layout()
plt.show()

