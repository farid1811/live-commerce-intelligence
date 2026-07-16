import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# =========================
# LOAD DATA
# =========================
df = pd.read_excel('Data Skripsi full.xlsx', sheet_name='Sheet1', header=1)

print("POLYNOMIAL REGRESSION (Degree=2) + SGD - ALL DATA")
print("="*60)

X_all = df[['Durasi_Jam', 'Penonton Aktif']].values
y_all = df['Produk Terjual'].values

print(f"Jumlah data: {len(X_all)}")
print(f"Jumlah data Produk Terjual = 0: {np.sum(y_all == 0)}")

# =========================
# POLYNOMIAL FEATURES
# =========================
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_all)

print(f"\nShape sebelum polynomial: {X_all.shape}")
print(f"Shape setelah polynomial: {X_poly.shape}")

# =========================
# NORMALISASI
# =========================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_poly_scaled = scaler_X.fit_transform(X_poly)
y_scaled = scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

# =========================
# SGD + CONSTRAINTS
# =========================
def sgd_polynomial_constrained(X, y, learning_rate=0.01, epochs=1000):
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
            
            # Constraints
            theta_new = np.maximum(theta_new, 0)
            bias_new = max(bias_new, 0)
            
            theta = theta_new
            bias = bias_new
            
    return theta, bias

print("\nTRAINING MODEL...")
theta_poly, bias_poly = sgd_polynomial_constrained(
    X_poly_scaled, y_scaled, learning_rate=0.01, epochs=1000
)

# =========================
# PREDIKSI
# =========================
y_pred_poly_scaled = np.dot(X_poly_scaled, theta_poly) + bias_poly
y_pred_poly = scaler_y.inverse_transform(
    y_pred_poly_scaled.reshape(-1, 1)
).flatten()

# =========================
# EVALUASI
# =========================
mae_poly = mean_absolute_error(y_all, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_all, y_pred_poly))

# R2
ss_res = np.sum((y_all - y_pred_poly) ** 2)
ss_tot = np.sum((y_all - np.mean(y_all)) ** 2)
r2_poly = 1 - (ss_res / ss_tot)

# =========================
# MAPE FIX (TANPA UBAH MODEL)
# =========================
mask_poly = y_all != 0
mape_poly = mean_absolute_percentage_error(
    y_all[mask_poly],
    y_pred_poly[mask_poly]
) * 100

# =========================
# KEMBALIKAN KOEFISIEN KE SKALA ASLI
# =========================
theta_poly_original = theta_poly / scaler_X.scale_
bias_poly_original = bias_poly - np.dot(
    scaler_X.mean_ / scaler_X.scale_,
    theta_poly
)
bias_poly_original = scaler_y.inverse_transform(
    [[bias_poly_original]]
)[0][0]

feature_names = poly.get_feature_names_out(
    ['Durasi_Jam', 'Penonton_Aktif']
)

# =========================
# OUTPUT HASIL
# =========================
print("\n" + "="*60)
print("HASIL POLYNOMIAL REGRESSION - ALL DATA")
print("="*60)

print("\nParameter Model:")
for coef, name in zip(theta_poly_original, feature_names):
    print(f"{name}: {coef:.6f}")
print(f"Intercept: {bias_poly_original:.6f}")

print("\nMetrik Evaluasi:")
print(f"R²   : {r2_poly:.4f}")
print(f"MAE  : {mae_poly:.4f}")
print(f"RMSE : {rmse_poly:.4f}")
print(f"MAPE : {mape_poly:.2f}%")

# =========================
# VISUALISASI
# =========================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(y_all, y_pred_poly, alpha=0.6)
plt.plot([y_all.min(), y_all.max()],
         [y_all.min(), y_all.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Polynomial)")

plt.subplot(1,2,2)
residual = y_all - y_pred_poly
plt.scatter(y_pred_poly, residual, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual Plot (Polynomial)")

plt.tight_layout()
plt.show()