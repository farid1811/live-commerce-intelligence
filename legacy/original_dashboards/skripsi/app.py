import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Dashboard Prediksi", layout="wide")

st.title("Dashboard Prediksi Penjualan Live Streaming")

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("Data Skripsi full.xlsx", sheet_name="Sheet1", header=1)

X = df[['Durasi_Jam', 'Penonton Aktif']].values
y = df['Produk Terjual'].values

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model_linear.pkl")

theta = model["theta"]
bias = model["bias"]

# =========================
# INPUT USER
# =========================
st.sidebar.header("Input")

durasi = st.sidebar.slider("Durasi Live (Jam)", 1.0, 24.0, 12.0)
penonton = st.sidebar.slider("Jumlah Penonton", 10, 500, 100)

# =========================
# PREDIKSI
# =========================
pred = theta[0]*durasi + theta[1]*penonton + bias

y_pred_all = theta[0]*X[:,0] + theta[1]*X[:,1] + bias

# =========================
# METRIK
# =========================
mae = mean_absolute_error(y, y_pred_all)
rmse = np.sqrt(mean_squared_error(y, y_pred_all))

ss_res = np.sum((y - y_pred_all)**2)
ss_tot = np.sum((y - np.mean(y))**2)

r2 = 1 - (ss_res/ss_tot)

# =========================
# OUTPUT
# =========================
st.subheader("Hasil Prediksi")

c1, c2, c3 = st.columns(3)

c1.metric("Durasi", f"{durasi} Jam")
c2.metric("Penonton", penonton)
c3.metric("Prediksi Penjualan", f"{pred:.2f}")

# =========================
# EVALUASI MODEL
# =========================
st.subheader("Evaluasi Model")

m1, m2, m3 = st.columns(3)

m1.metric("R²", f"{r2:.4f}")
m2.metric("MAE", f"{mae:.2f}")
m3.metric("RMSE", f"{rmse:.2f}")

# =========================
# VISUALISASI
# =========================
st.subheader("Visualisasi Model")

col1, col2 = st.columns(2)

# Actual vs Predicted
fig1, ax1 = plt.subplots()

ax1.scatter(y, y_pred_all, alpha=0.6)
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')

ax1.set_xlabel("Actual")
ax1.set_ylabel("Predicted")
ax1.set_title("Actual vs Predicted")

# Residual Plot
fig2, ax2 = plt.subplots()

residual = y - y_pred_all

ax2.scatter(y_pred_all, residual, alpha=0.6)
ax2.axhline(0, color='red', linestyle='--')

ax2.set_xlabel("Predicted")
ax2.set_ylabel("Residual")
ax2.set_title("Residual Plot")

with col1:
    st.pyplot(fig1)

with col2:
    st.pyplot(fig2)

st.caption("Dashboard Implementasi Model Regresi Linear (SGD)")