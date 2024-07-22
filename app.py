import streamlit as st
import tensorflow as tf
from helper import predict_image
from PIL import Image

model = tf.keras.models.load_model(r'Model\model.h5')
class_names = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

st.set_page_config(page_title ='Plant Seedling Image Classification', page_icon = '🌱', layout ='wide')
st.markdown("<h1 style = 'text-align: center;'>Plant Seedling Image Classification 🌱</h1>", unsafe_allow_html=True)

st.write("Instructions: Upload an image to see the predicted class!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("")
    st.write("Classifying...")

    predicted_class, probabilities = predict_image(model, image, class_names)

    st.write(f'Predicted class: {predicted_class}')