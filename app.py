import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.title("COVID-19 LSTM Vaka Tahmini")

model = tf.keras.models.load_model('covid_lstm_cnn_model.h5', compile=False)
scaler_min = np.load('scaler_min.npy')
scaler_scale = np.load('scaler_scale.npy')

df = pd.read_csv('dataset/day_wise.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
data = df[['New cases']].copy()


st.subheader("Günlük Yeni Vaka Sayısı (Tüm Veri)")
plt.figure(figsize=(10,5))
plt.plot(data.index, data['New cases'])
plt.title('Günlük Yeni Vaka Sayısı')
plt.xlabel('Tarih')
plt.ylabel('Yeni Vaka')
plt.grid()
st.pyplot(plt)

scaler = MinMaxScaler()
scaler.min_ = scaler_min
scaler.scale_ = scaler_scale

data_scaled = scaler.transform(data)

window_size = 10
current_batch = data_scaled[-window_size:].reshape((1, window_size, 1))
future_steps = 7
future_predictions = []

for i in range(future_steps):
    pred = model.predict(current_batch, verbose=0)[0]
    future_predictions.append(pred)

    current_batch = np.append(current_batch[:,1:,:], [[pred]], axis=1)

future_predictions_inv = scaler.inverse_transform(future_predictions)


st.subheader("7 Günlük İleri Tahmin")
st.write(future_predictions_inv.flatten())

plt.figure(figsize=(8,5))
plt.plot(range(1, future_steps+1), future_predictions_inv.flatten(), marker='o')
plt.title('7 Günlük Gelecek Tahmini')
plt.xlabel('Günler')
plt.ylabel('Yeni Vaka Sayısı')
plt.grid()
st.pyplot(plt)
