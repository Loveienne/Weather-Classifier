import streamlit as st
import numpy as np
import tensorflow
import h5py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image, ImageOps
import os

# Load model
model_path = 'weather_classifier_model4.h5'
# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Check if the file is a valid HDF5 file
try:
    with h5py.File(model_path, 'r') as f:
        print("File is a valid HDF5 file.")
except OSError as e:
    print(f"File is not a valid HDF5 file or is corrupted: {e}")

model = load_model(model_path)
print(f"Model {model_path} exists")

# Function to predict
def predict(img):
    image = ImageOps.fit(img, (256, 256))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
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
    prediction = predict(image)
    
    try:
        st.write(f"Image is a {prediction}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
