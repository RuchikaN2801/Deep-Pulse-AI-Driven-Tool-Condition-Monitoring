import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import sys

# Fix import path
sys.path.append('../src')

from hybrid_model import CNN_LSTM_Model

st.title("🔧 Deep-Pulse: Tool Health Monitoring")

# Load model
model = CNN_LSTM_Model(num_classes=3)
model.load_state_dict(torch.load("../models/saved_models/cnn_lstm.pth", map_location='cpu'))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Upload image
uploaded_file = st.file_uploader("Upload a Spectrogram Image")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Spectrogram")

    img = transform(image).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, 1).item()

    classes = ["Healthy", "Worn", "Failure"]
    st.success(f"Prediction: {classes[pred]}")
