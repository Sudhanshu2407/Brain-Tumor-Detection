import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('brain_tumor_model.h5')  # Update this with your model's path

# Define the class names
#{'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
class_names = ['glioma','meningioma','No Tumor', 'pituitary']

# Function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))  # Ensure the image size matches the model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to make predictions
def predict_image(uploaded_file):
    img_array = preprocess_image(uploaded_file)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return predicted_class, confidence

# Streamlit App
st.title("TumorScan: Brain Tumor Detection")
st.write("Upload an MRI image to determine whether it contains a brain tumor.")

# Image upload section
uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=True)

    # Make prediction
    predicted_class, confidence = predict_image(uploaded_file)

    # Display prediction
    st.write(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}**")
