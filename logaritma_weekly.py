import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# Load data
df = pd.read_excel('Data Skripsi full.xlsx', sheet_name='Sheet1', header=1)

print("LOGARITHMIC REGRESSION + SGD - DATA WEEKLY")
print("="*60)

# DATA WEEKLY
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

print(f"Data Weekly: {len(df_weekly)} samples")

# Persiapan data
X_weekly = df_weekly[['Durasi_Jam', 'Penonton Aktif']].values
y_weekly = df_weekly['Produk Terjual'].values

# Apply logarithmic transformation to features (avoid log(0) by adding small constant)
X_log = np.log(X_weekly + 1)  # +1 to avoid log(0)

print(f"Shape sebelum log: {X_weekly.shape}")
print(f"Shape setelah log: {X_log.shape}")
print(f"Sample data asli: {X_weekly[:3]}")
print(f"Sample data log: {X_log[:3]}")

# Normalisasi
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_log_scaled = scaler_X.fit_transform(X_log)
y_scaled = scaler_y.fit_transform(y_weekly.reshape(-1, 1)).flatten()

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
mae_log = mean_absolute_error(y_weekly, y_pred_log)
mape_log = mean_absolute_percentage_error(y_weekly, y_pred_log)
rmse_log = np.sqrt(mean_squared_error(y_weekly, y_pred_log))

# Calculate R²
ss_res = np.sum((y_weekly - y_pred_log) ** 2)
ss_tot = np.sum((y_weekly - np.mean(y_weekly)) ** 2)
r2_log = 1 - (ss_res / ss_tot)

print("\n" + "="*50)
print("HASIL LOGARITHMIC REGRESSION - DATA WEEKLY")
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

# Perbandingan dengan Linear Regression
print(f"\n" + "="*60)
print("PERBANDINGAN LINEAR vs LOGARITHMIC REGRESSION (DATA WEEKLY)")
print("="*60)
print(f"{'Metric':<10} {'Linear':<12} {'Logarithmic':<12} {'Improvement':<12}")
print(f"{'-'*60}")
print(f"{'R²':<10} {0.5469:<12.4f} {r2_log:<12.4f} {r2_log - 0.5469:>+10.4f}")
print(f"{'MAE':<10} {6.0432:<12.4f} {mae_log:<12.4f} {6.0432 - mae_log:>+10.4f}")
print(f"{'RMSE':<10} {8.1847:<12.4f} {rmse_log:<12.4f} {8.1847 - rmse_log:>+10.4f}")

if r2_log > 0.5469 and mae_log < 6.0432:
    print(f"\n🎉 LOGARITHMIC LEBIH BAIK dari Linear Regression!")
elif r2_log < 0.5469 and mae_log > 6.0432:
    print(f"\n❌ LOGARITHMIC LEBIH BURUK dari Linear Regression!")
else:
    print(f"\n📊 LOGARITHMIC dan Linear memiliki trade-off yang berbeda")

# Visualisasi perbandingan
plt.figure(figsize=(15, 10))

# Data Linear Weekly dari sebelumnya
y_pred_linear = np.array([23.7004, 24.7167, 21.3332, 22.0143, 40.6039, 32.0857, 35.4251, 
                          29.7684, 34.8185, 33.6705, 35.7746, 33.6221, 43.0762, 42.5174, 
                          46.3780, 46.3780, 46.3780, 46.3780, 46.3780, 46.3780, 46.3780, 
                          46.3780, 46.3780, 46.3780, 46.3780, 46.3780, 46.3780, 46.3780, 
                          46.3780, 46.3780, 46.3780, 46.3780, 46.3780, 46.3780, 46.3780, 
                          46.3780, 46.3780])

# Plot 1: Actual vs Predicted Comparison
plt.subplot(2, 3, 1)
plt.scatter(y_weekly, y_pred_linear, alpha=0.6, s=80, color='blue', label='Linear')
plt.scatter(y_weekly, y_pred_log, alpha=0.6, s=80, color='orange', label='Logarithmic')
plt.plot([y_weekly.min(), y_weekly.max()], [y_weekly.min(), y_weekly.max()], 'k--', lw=1)
plt.xlabel('Actual Produk Terjual')
plt.ylabel('Predicted Produk Terjual')
plt.title('Perbandingan: Linear vs Logarithmic\n(Data Weekly)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Loss convergence
plt.subplot(2, 3, 2)
plt.plot(range(0, 1000, 100), losses_log, marker='o', color='orange')
plt.xlabel('Epoch (x100)')
plt.ylabel('Loss (MSE)')
plt.title('Konvergensi SGD\n(Logarithmic Regression)')
plt.grid(True, alpha=0.3)

# Plot 3: Residuals comparison
plt.subplot(2, 3, 3)
residuals_linear = y_weekly - y_pred_linear
residuals_log = y_weekly - y_pred_log

plt.scatter(y_pred_linear, residuals_linear, alpha=0.6, s=60, color='blue', label='Linear')
plt.scatter(y_pred_log, residuals_log, alpha=0.6, s=60, color='orange', label='Logarithmic')
plt.axhline(y=0, color='k', linestyle='--', lw=1)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Error comparison per sample
plt.subplot(2, 3, 4)
samples = range(len(y_weekly))
errors_linear = np.abs(y_weekly - y_pred_linear)
errors_log = np.abs(y_weekly - y_pred_log)

plt.plot(samples, errors_linear, marker='o', color='blue', label='Linear Error', alpha=0.7)
plt.plot(samples, errors_log, marker='s', color='orange', label='Logarithmic Error', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Absolute Error')
plt.title('Error per Sample\n(Lower is Better)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Performance metrics comparison
plt.subplot(2, 3, 5)
metrics = ['MAE', 'RMSE']
linear_values = [6.0432, 8.1847]
log_values = [mae_log, rmse_log]

x_pos = np.arange(len(metrics))
width = 0.35

plt.bar(x_pos - width/2, linear_values, width, label='Linear', alpha=0.7, color='blue')
plt.bar(x_pos + width/2, log_values, width, label='Logarithmic', alpha=0.7, color='orange')
plt.ylabel('Error Values')
plt.title('Performance Metrics Comparison')
plt.xticks(x_pos, metrics)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Tampilkan beberapa prediksi
print(f"\n" + "="*50)
print("SAMPLE PREDIKSI LOGARITHMIC (10 data pertama)")
print("="*50)

sample_results = pd.DataFrame({
    'Bulan': df_weekly['Bulan_Str'].head(10),
    'Durasi_Jam': df_weekly['Durasi_Jam'].head(10),
    'Penonton_Aktif': df_weekly['Penonton Aktif'].head(10),
    'Actual': y_weekly[:10],
    'Pred_Linear': y_pred_linear[:10],
    'Pred_Logarithmic': y_pred_log[:10],
    'Error_Linear': np.abs(y_weekly[:10] - y_pred_linear[:10]),
    'Error_Log': np.abs(y_weekly[:10] - y_pred_log[:10])
})
print(sample_results.round(4))

print(f"\n📊 INTERPRETASI LOGARITHMIC MODEL:")
print(f"- Model menangkap hubungan non-linear antara variabel input dan output")
print(f"- Koefisien menunjukkan sensitivitas terhadap perubahan persentase")
print(f"- Cocok untuk data dengan diminishing returns")