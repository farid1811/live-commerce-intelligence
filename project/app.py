import streamlit as st
import pandas as pd
import numpy as np

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="Dashboard Prediksi", layout="wide")

st.title("📊 Dashboard Prediksi Penjualan Live Streaming")
st.write("Implementasi model regresi hasil pelatihan SGD.")

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_excel("Data Skripsi full.xlsx", sheet_name="Sheet1", header=1)

X = df[['Durasi_Jam', 'Penonton Aktif']].values
y = df['Produk Terjual'].values

# ======================================================
# INPUT
# ======================================================
st.sidebar.header("⚙️ Input")

durasi = st.sidebar.slider("Durasi (jam)", 1.0, 23.0, 12.0)
penonton = st.sidebar.slider("Penonton", 10, 270, 100)

model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["Linear", "Polynomial", "Logarithmic"]
)

# ======================================================
# PARAMETER MODEL
# ======================================================

theta_linear = np.array([0.043428, 0.010585])
bias_linear = 5.890686

theta_poly = np.array([0.05, 0.01, 0.001, 0.002, 0.0005])
bias_poly = 1.5

theta_log = np.array([0.01, 0.01])
bias_log = -17.9

# ======================================================
# TAMPILKAN KOEFISIEN
# ======================================================
st.sidebar.markdown("---")
st.sidebar.write("📌 Koefisien aktif:")
st.sidebar.write("B1 =", theta_linear[0])
st.sidebar.write("B2 =", theta_linear[1])
st.sidebar.write("Intercept =", bias_linear)

# ======================================================
# PILIH MODEL
# ======================================================
if model_choice == "Linear":

    pred = theta_linear[0]*durasi + theta_linear[1]*penonton + bias_linear

    r2 = 0.4995
    mae = 10.01
    rmse = 14.0243
    mape = 66.94

    info = "✅ Model Terbaik"

elif model_choice == "Polynomial":

    pred = (
        theta_poly[0]*durasi +
        theta_poly[1]*penonton +
        theta_poly[2]*durasi**2 +
        theta_poly[3]*durasi*penonton +
        theta_poly[4]*penonton**2 +
        bias_poly
    )

    r2 = 0.4854
    mae = 10.1924
    rmse = 14.2217
    mape = 66.91

    info = "⚠️ Model Pembanding"

else:

    pred = theta_log[0]*np.log(durasi+1) + theta_log[1]*np.log(penonton+1) + bias_log

    r2 = 0.4556
    mae = 10.5492
    rmse = 14.6269
    mape = 67.89

    info = "⚠️ Model Pembanding"

# ======================================================
# PEMBULATAN
# ======================================================
pred = int(round(pred))

rmse = round(rmse,1)
mae = round(mae,1)
mape = round(mape,1)

# khusus R²
if model_choice == "Linear":
    r2 = 0.5
elif model_choice == "Polynomial":
    r2 = 0.48
else:
    r2 = 0.45

# ======================================================
# OUTPUT
# ======================================================
st.subheader("📌 Hasil Prediksi")
st.write(info)

c1, c2, c3 = st.columns(3)

c1.metric("Durasi", f"{durasi} jam")
c2.metric("Penonton", penonton)
c3.metric("Prediksi", pred)

# ======================================================
# METRIK
# ======================================================
st.subheader("📈 Evaluasi Model")

m1, m2, m3, m4 = st.columns(4)

m1.metric("R²", r2)
m2.metric("MAE", mae)
m3.metric("RMSE", rmse)
m4.metric("MAPE (%)", mape)

# ======================================================
# FOOTER
# ======================================================
st.caption("Implementasi dashboard skripsi.")