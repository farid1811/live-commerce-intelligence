import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# =========================
# LOAD DATA
# =========================
df = pd.read_excel('Data Skripsi full.xlsx', sheet_name='Sheet1', header=1)

print("LOGARITHMIC REGRESSION + SGD - ALL DATA")
print("="*60)

X_all = df[['Durasi_Jam', 'Penonton Aktif']].values
y_all = df['Produk Terjual'].values

print(f"Jumlah data: {len(X_all)}")
print(f"Jumlah data Produk Terjual = 0: {np.sum(y_all == 0)}")

# =========================
# LOG TRANSFORMATION
# =========================
X_log = np.log(X_all + 1)  # +1 untuk hindari log(0)

# =========================
# NORMALISASI
# =========================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_log_scaled = scaler_X.fit_transform(X_log)
y_scaled = scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

# =========================
# SGD + CONSTRAINTS
# =========================
def sgd_log_constrained(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    bias = 0.1

    for epoch in range(epochs):
        for i in range(m):
            y_pred = np.dot(X[i], theta) + bias
            error = y_pred - y[i]

            grad_theta = error * X[i]
            grad_bias = error

            theta_new = theta - learning_rate * grad_theta
            bias_new = bias - learning_rate * grad_bias

            # Constraint: koefisien tidak boleh negatif
            theta = np.maximum(theta_new, 0)
            bias = max(bias_new, 0)

    return theta, bias

print("\nTRAINING MODEL...")
theta_log, bias_log = sgd_log_constrained(
    X_log_scaled, y_scaled, learning_rate=0.01, epochs=1000
)

# =========================
# PREDIKSI
# =========================
y_pred_scaled = np.dot(X_log_scaled, theta_log) + bias_log
y_pred_log = scaler_y.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).flatten()

# =========================
# EVALUASI
# =========================
mae_log = mean_absolute_error(y_all, y_pred_log)
rmse_log = np.sqrt(mean_squared_error(y_all, y_pred_log))

# R2
ss_res = np.sum((y_all - y_pred_log) ** 2)
ss_tot = np.sum((y_all - np.mean(y_all)) ** 2)
r2_log = 1 - (ss_res / ss_tot)

# =========================
# MAPE FIX (VALID)
# =========================
mask_log = y_all != 0
mape_log = mean_absolute_percentage_error(
    y_all[mask_log],
    y_pred_log[mask_log]
) * 100

# =========================
# KEMBALIKAN KOEFISIEN KE SKALA ASLI
# =========================
theta_log_original = theta_log / scaler_X.scale_
bias_log_original = bias_log - np.dot(
    scaler_X.mean_ / scaler_X.scale_,
    theta_log
)
bias_log_original = scaler_y.inverse_transform(
    [[bias_log_original]]
)[0][0]

# =========================
# OUTPUT HASIL
# =========================
print("\n" + "="*60)
print("HASIL LOGARITHMIC REGRESSION - ALL DATA")
print("="*60)

print("\nParameter Model:")
print(f"Koef log(Durasi+1): {theta_log_original[0]:.6f}")
print(f"Koef log(Penonton+1): {theta_log_original[1]:.6f}")
print(f"Intercept: {bias_log_original:.6f}")

print("\nMetrik Evaluasi:")
print(f"R²   : {r2_log:.4f}")
print(f"MAE  : {mae_log:.4f}")
print(f"RMSE : {rmse_log:.4f}")
print(f"MAPE : {mape_log:.2f}%")

print("\nModel Logarithmic:")
print(f"Produk_Terjual = {theta_log_original[0]:.6f} × log(Durasi+1) + "
      f"{theta_log_original[1]:.6f} × log(Penonton+1) + "
      f"{bias_log_original:.6f}")

# =========================
# VISUALISASI
# =========================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(y_all, y_pred_log, alpha=0.6)
plt.plot([y_all.min(), y_all.max()],
         [y_all.min(), y_all.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Logarithmic)")

plt.subplot(1,2,2)
residual = y_all - y_pred_log
plt.scatter(y_pred_log, residual, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual Plot (Logarithmic)")

plt.tight_layout()
plt.show()