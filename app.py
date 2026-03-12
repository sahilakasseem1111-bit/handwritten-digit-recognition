import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("digit_model.h5")

st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Upload a digit image", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("L")
    img = img.resize((28,28))

    img = np.array(img)
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1,28,28)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    st.write("Predicted Digit:", digit)