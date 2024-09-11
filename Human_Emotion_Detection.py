import os
import streamlit as st
import subprocess
import sys

# Install dependencies dynamically
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_package('torch')
install_package('PIL')
install_package('torchvision')


import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

# Define the path to the model file
MODEL_PATH = 'Human_Emotion_Detection.pt'

# Load the model
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Define image transformation
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Change to match model's input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

# Get emotion label
def predict_emotion(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        logits = output.logits  # Extract logits if available
        predicted_class = torch.argmax(logits, dim=1).item()
        probabilities = torch.softmax(logits, dim=1)
    return predicted_class, probabilities

# Emotion classes (based on FER-2013 dataset)
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit app layout
st.title("Human Emotion Detection")

st.write("""
This app uses a deep learning model to detect human emotions from an image.
Upload an image and the app will predict the emotion.
""")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Transform and predict
    image_tensor = transform_image(image)
    model = load_model(MODEL_PATH)

    if model is not None and st.button('Predict Emotion'):
        with st.spinner('Predicting...'):
            predicted_class = predict_emotion(model, image_tensor)

            # Show result
            st.write(f"{emotion}: {probabilities[0][i].item()*100:.2f}%")
