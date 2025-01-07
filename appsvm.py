import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Muat model yang sudah disimpan
model = joblib.load('fish_svm_model.pkl')

# Tampilan utama aplikasi
st.title("Prediksi Spesies Ikan menggunakan SVM")

# Input untuk fitur prediksi
length = st.number_input("Panjang (cm)", min_value=0.0, step=0.1)
weight = st.number_input("Berat (gram)", min_value=0.0, step=0.1)
w_l_ratio = st.number_input("Rasio Berat-Panjang", min_value=0.0, step=0.1)

# Prediksi berdasarkan input
if st.button("Prediksi"):
    if length > 0 and weight > 0 and w_l_ratio > 0:
        # Sesuaikan kolom input dengan model
        input_data = pd.DataFrame([[length, weight, w_l_ratio]], columns=['length', 'weight', 'w_l_ratio'])
        try:
            prediction = model.predict(input_data)
            st.success(f'Prediksi Nama Ikan: {prediction[0]}')
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
    else:
        st.warning("Harap isi semua input sebelum melakukan prediksi!")
