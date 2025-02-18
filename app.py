import streamlit as st
import pickle
import numpy as np

# Import the model and the dataframe
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predictor")

# Brand selection
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# RAM selection
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight input
weight = st.number_input('Weight of the Laptop (in kg)', min_value=0.5, max_value=10.0, step=0.1)

# Touchscreen option
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS display option
ips = st.selectbox('IPS Display', ['No', 'Yes'])

# Screen size slider
screen_size = st.slider('Screen size (in inches)', 10.0, 18.0, 13.0)

# Screen resolution selection
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# CPU selection
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# HDD selection
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD selection
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU selection
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# OS selection
os = st.selectbox('Operating System', df['os'].unique())

if st.button('Predict Price'):
    # Preprocessing inputs
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI (Pixels Per Inch)
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = np.sqrt(X_res**2 + Y_res**2) / screen_size

    # Prepare the query for prediction
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, -1)

    # Predict the price
    predicted_price = np.exp(pipe.predict(query)[0])  # Inverse log transformation if needed

    # Show the predicted price
    st.title(f"The predicted price of this laptop configuration is ₹{int(predicted_price)}")
