import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your TFLite model
interpreter = tf.lite.Interpreter(model_path="fire_detection.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input details for debugging
print("Input Details:", input_details)
print("Output Details:", output_details)

# Function to preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (150, 150))  # Resize to match the model input size
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 150, 150, 3)
    
    # Ensure the image has 3 channels (RGB). Remove alpha channel if present.
    if image.shape[-1] == 4:
        image = image[..., :3]  # Remove alpha channel
    
    image = image.astype(np.float32) / 255.0  # Normalize the image to [0, 1]
    return image


# Function to predict fire
def detect_fire(image):
    preprocessed_image = preprocess_image(image)
    print("Preprocessed Image Shape:", preprocessed_image.shape)  # Debugging
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image.astype(np.float32))
    
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Model Output:", output_data)  # Debugging
    
    return output_data[0][0]  # Assuming it's a single probability

# Streamlit interface
st.title("Over4Fire - Fire Detection Web App")

st.write("Upload an image or use the camera for fire detection.")

# Upload image or take a picture
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Run fire detection
    prediction = detect_fire(img_array)

    if prediction > 0.5:
        st.write("🔥 FIRE risk!")
    else:
        st.write("✅ No Fire risk.")

# Optional: Camera input (using OpenCV, but only works locally)
if st.button("Use Camera"):
    camera = cv2.VideoCapture(0)  # Open the webcam
    
    # Adjust camera settings if needed (exposure, brightness)
    camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)  # Adjust brightness (range between 0-1)
    camera.set(cv2.CAP_PROP_EXPOSURE, -6)  # Adjust exposure (lower value makes it brighter)
    
    ret, frame = camera.read()
    if ret:
        st.image(frame, channels="BGR")
        prediction = detect_fire(frame)
        if prediction > 0.5:
            st.write("🔥 Fire Detected!")
        else:
            st.write("✅ No Fire Detected.")
    camera.release()

