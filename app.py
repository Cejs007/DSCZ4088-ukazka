import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

def preprocess_image(img, target_size=(128, 128)):  # velikost přizpůsob modelu
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # přidáme kanál
    return np.expand_dims(img_array, axis=0)  # přidáme batch dimenzi

# Načti model
model = tf.keras.models.load_model('model.keras')  # uprav podle názvu souboru

# GUI
st.title("🧠 Predikce Obrázku pomocí Keras Modelu")

uploaded_file = st.file_uploader("📁 Vyber obrázek", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # 'L' znamená grayscale
    st.image(image, caption='📸 Nahraný obrázek', use_column_width=True)

    if st.button("🔍 Spustit predikci"):
        input_data = preprocess_image(image)
        prediction = model.predict(input_data)
        st.write("📊 Výsledek predikce:", prediction)