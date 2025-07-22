import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Page configuration
st.set_page_config(page_title="Pollen's Profiling", layout="centered")
st.title("🌼 Pollen's Profiling - Image Classifier")
st.write("📸 Upload a pollen grain image to get its predicted class.")

# Load the trained model
model = load_model("pollen_model.h5")
class_names = ['Class_1', 'Class_2', 'Class_3']  # ✅ Replace with actual class names

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Prediction logic
if uploaded_file:
    # Read and decode the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Show uploaded image
    st.image(image, caption="Uploaded Image", channels="BGR", use_container_width=True)
# Replace these with meaningful descriptions for your pollen classes
    class_descriptions = {
        'Class_1': '🌿 Description for Class 1: Found in seasonal crops, oval-shaped structure, moderate allergy risk.',
        'Class_2': '🌸 Description for Class 2: Belongs to flowering trees, round-shaped, high allergy potential.',
        'Class_3': '🌾 Description for Class 3: Associated with grass family, lightweight, spreads easily in wind.'
}

    if st.button("🔍 Classify"):
        # Preprocess the image
        resized = cv2.resize(image, (64, 64)) / 255.0
        input_data = np.expand_dims(resized, axis=0)

        # Predict
        prediction = model.predict(input_data)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Show result
        st.success(f"✅ **Predicted Class:** {predicted_class}")
        st.info(f"📊 Confidence: {confidence*100:.2f}%")
        # Show description if available
        description = class_descriptions.get(predicted_class, "No description available.")
        st.write(f"📌 **Description:** {description}")