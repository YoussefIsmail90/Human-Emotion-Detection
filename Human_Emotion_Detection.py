import streamlit as st
import subprocess
import sys

# Install torch dynamically
def install_torch():
    try:
        import torch
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==1.10.0"])

install_torch()

# After ensuring torch is installed, proceed with the rest of your app
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

# Load the model
@st.cache_resource
def load_model():
    model = torch.load('models/best_vit_fer2013_model_Human_Emotion_Detection.pt', map_location=torch.device('cpu'))
    model.eval()  # Set the model to evaluation mode
    return model

# Define image transformation
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),  # Resize the image to match model's expected size
        transforms.Grayscale(),  # Convert to grayscale (FER2013 dataset is grayscale)
        transforms.ToTensor(),  # Convert image to Tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize image
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

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
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Transform and predict
    image_tensor = transform_image(image)
    model = load_model()

    if st.button('Predict Emotion'):
        predicted_class, probabilities = predict_emotion(model, image_tensor)

        # Show result
        st.write(f"Predicted Emotion: {emotion_classes[predicted_class]}")
        
        # Show probabilities
        st.write("Emotion Probabilities:")
        for i, emotion in enumerate(emotion_classes):
            st.write(f"{emotion}: {probabilities[0][i].item()*100:.2f}%")
