import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image, ImageOps
import os

# Load model
model_path = 'weather_classifier_model.h5'
print(f"Loading model from: {model_path}")

try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Function to predict
def predict(image):
    image = ImageOps.fit(image, (256, 256))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    prediction = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)[0]
    class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
  
    return class_names[class_index]    

# Streamlit app
st.title("Weather Classifier")
st.write("Upload an image to classify")

file_upload = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "png"])


if file_upload is not None:
    image = Image.open(file_upload)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    prediction = predict(image)
    
    try:
        st.write(f"Image is a {prediction}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
