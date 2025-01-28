import streamlit as st
import pandas as pd
import pickle
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Fungsi untuk memuat model Prophet
def load_prophet_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Fungsi untuk membuat prediksi menggunakan Prophet
def predict_with_prophet(model, periods, cap_value=None):
    # Membuat dataframe untuk masa depan
    future = model.make_future_dataframe(periods=periods)
    
    # Menambahkan kolom cap jika pertumbuhan logistic digunakan
    if cap_value:
        future['cap'] = cap_value  # Tetapkan kapasitas maksimum untuk semua baris
    
    # Membuat prediksi
    forecast = model.predict(future)
    return forecast

# Fungsi untuk evaluasi model
def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# Streamlit Title
st.title("Prediksi Harga Saham Tesla Menggunakan Prophet")

# Sidebar untuk parameter
st.sidebar.header("Pengaturan")
dataset_path = "TESLA.csv" # Path tetap untuk dataset
model_path = "best_prophet_model.pkl"

# Slider untuk jumlah tahun prediksi ke depan
prediction_years = st.sidebar.slider("Jumlah Tahun Prediksi ke Depan", 1, 5, 1)  # Input dalam tahun
prediction_days = prediction_years * 365  # Konversi tahun ke hari

# Sidebar tambahan untuk pengaturan logistic growth
st.sidebar.subheader("Pengaturan Logistic Growth")
cap_value = None

# Memuat dataset pengguna
st.subheader("Data Historis Saham Tesla")
try:
    stock_data = pd.read_csv(dataset_path)
    if 'Unnamed: 0' in stock_data.columns:
        stock_data = stock_data.drop(columns=['Unnamed: 0'])
        
    if 'Date' in stock_data.columns and 'Close' in stock_data.columns:
        stock_data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
        
    stock_data['ds'] = pd.to_datetime(stock_data['ds'])  # Pastikan kolom ds adalah datetime
    stock_data = stock_data.sort_values(by='ds') 
    st.write("Dataset yang dimuat:")
    st.dataframe(stock_data)

    # Tambahkan kapasitas maksimum jika logistic growth digunakan
    cap_margin = st.sidebar.slider("Margin Kapasitas Maksimum (%)", 10, 50, 20)
    cap_value = stock_data['y'].max() * (1 + cap_margin / 100)
    stock_data['cap'] = cap_value
    st.sidebar.write(f"Kapasitas Maksimum (cap): {cap_value:.2f}")

    # Visualisasi data historis
    st.line_chart(stock_data.set_index('ds')['y'])

except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat dataset: {e}")

# Memuat model Prophet dan melakukan prediksi
st.subheader("Prediksi Harga Saham Tesla")
if st.sidebar.button("Prediksi"):
    try:
        # Muat model Prophet
        model = load_prophet_model(model_path)
        
        # Lakukan prediksi
        forecast = predict_with_prophet(model, prediction_days, cap_value) #if use_logistic else None)

        # Visualisasi hasil prediksi
        st.write("Hasil Prediksi:")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(stock_data['ds'], stock_data['y'], 'k.', label="Data Historis")
        ax.plot(forecast['ds'], forecast['yhat'], 'b-', label="Prediksi")
        ax.fill_between(
            forecast['ds'], 
            forecast['yhat_lower'], 
            forecast['yhat_upper'], 
            color='blue', alpha=0.2, label="Batas Atas/Bawah (yhat_lower/yhat_upper)"
        )
        ax.set_title("Grafik Hasil Prediksi")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga Saham")
        ax.legend()
        st.pyplot(fig)

        # Tampilkan data prediksi dalam tabel
        st.write("Tabel Hasil Prediksi:")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prediction_days))
        
        # Evaluasi model
        st.subheader("Hasil Evaluasi Model")
        common_dates = stock_data.merge(forecast, on='ds', how='inner')  # Pastikan data historis cocok dengan prediksi
        actual = common_dates['y']
        predicted = common_dates['yhat']
        mae, mse, rmse = evaluate_model(actual, predicted)

        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")