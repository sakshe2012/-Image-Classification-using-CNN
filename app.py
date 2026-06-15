import streamlit as st
import numpy as np

from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("model.keras")

st.title("Handwritten Digit Classifier")

uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    # Convert to grayscale
    image = image.convert("L")

    # Resize to MNIST size
    image = image.resize((28, 28))

    image = np.array(image)

    image = image / 255.0

    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image)

    digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(f"Predicted Digit: {digit}")

    st.info(f"Confidence: {confidence:.2f}%")