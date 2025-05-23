import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Use st.cache_resource to cache the model loading
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("vegetable_classifier_model_final2.h5")
    return model

model = load_model()

# Load the class indices JSON file and create an index-to-class mapping.
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
index_to_class = {int(v): k for k, v in class_indices.items()}

st.title("Vegetable Classifier")
st.write("Upload an image from your computer to predict its vegetable class.")

# Show the supported vegetable classes as a note
st.markdown("### ℹ️ Supported Vegetable Classes:")
st.markdown(
    "- Bean\n"
    "- Bitter Gourd\n"
    "- Bottle Gourd\n"
    "- Brinjal\n"
    "- Broccoli\n"
    "- Cabbage\n"
    "- Capsicum\n"
    "- Carrot\n"
    "- Cauliflower\n"
    "- Cucumber\n"
    "- Papaya\n"
    "- Potato\n"
    "- Pumpkin\n"
    "- Radish\n"
    "- Tomato"
)

# Only allow file upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

if image is not None:
    if st.button("Predict"):
        # Preprocess the image: resize, normalize, and add a batch dimension
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)

        # If the image has an alpha channel, remove it by keeping only the first three channels
        if img_array.ndim == 3 and img_array.shape[-1] == 4:
            img_array = img_array[..., :3]

        img_array = img_array.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class] * 100

        # Get the predicted class label using the index_to_class dictionary
        predicted_label = index_to_class.get(predicted_class, "Unknown")
        st.success(f"Predicted Vegetable Class: **{predicted_label}** ({confidence:.2f}% confidence)")
