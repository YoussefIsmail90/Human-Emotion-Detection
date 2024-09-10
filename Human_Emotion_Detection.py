import os
import streamlit as st
import subprocess
import sys
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import timm

# Install dependencies dynamically
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_package('torch')
install_package('Pillow')  # Use Pillow for compatibility
install_package('torchvision')
install_package('timm') 

# Define your model architecture
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the layers of your model here
        # Example:
        self.vit = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=False)
        self.fc = torch.nn.Linear(768, 7)  # Adjust based on your model's final layer

    def forward(self, x):
        x = self.vit(x)
        x = self.fc(x)
        return x

# Define the path to the model file
MODEL_PATH = 'best_vit_fer2013_model_Human_Emotion_Detection.pt'

# Load the model
@st.cache_resource
def load_model(model_path):
    model = MyModel()
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Define image transformation
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

# Get emotion label
def predict_emotion(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
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
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

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
            predicted_class, probabilities = predict_emotion(model, image_tensor)

            # Show result
            st.write(f"Predicted Emotion: {emotion_classes[predicted_class]}")
            
            # Show probabilities
            st.write("Emotion Probabilities:")
            for i, emotion in enumerate(emotion_classes):
                st.write(f"{emotion}: {probabilities[0][i].item()*100:.2f}%")
