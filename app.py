from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import load_model

df = pd.read_csv('signnames.csv')

model = load_model('tsr_model.keras')

# st.write(df['SignName'])

st.title('Traffic Sign Recognition')

file = st.file_uploader('Choose an image', type='png')

data = []

if file is not None:
    img = Image.open(file)
    img = img.resize((30,30))
    img = np.expand_dims(img, axis=0)
    img = np.array(img)
    data.append(img)
    col1, col2 = st.columns([2, 1])
    col1.image(img, caption='Uploaded Image', use_column_width='True')

    if col2.button('Classify'):
        pred = model.predict(data[0])
        pred_classes = np.argmax(pred, axis=1)
        value = df['SignName'][pred_classes[0]]
        st.header(value)

