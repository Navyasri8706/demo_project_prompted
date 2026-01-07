import streamlit as st
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Human Face Detection App",
    page_icon="🙂",
    layout="centered"
)

st.title("🧠 Human Face Detection using Streamlit")
st.write("Upload an image and the app will detect **human faces** automatically.")

# -----------------------------
# Load Haar Cascade Model
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.subheader("📷 Uploaded Image Preview")
    st.image(image, use_column_width=True)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # Face Detection Parameters
    # -----------------------------
    scale_factor = st.slider(
        "Scale Factor (Detection Sensitivity)",
        min_value=1.05,
        max_value=1.5,
        value=1.1,
        step=0.05
    )

    min_neighbors = st.slider(
        "Min Neighbors (Accuracy Control)",
        min_value=3,
        max_value=10,
        value=5
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors
    )

    # Draw rectangles and labels
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img_array,
            "Human face identified",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    st leaving

    st.subheader("✅ Face Detection Result")
    st.image(img_array, use_column_width=True)

    st.success(f"Total Faces Detected: {len(faces)}")

else:
    st.info("👆 Please upload an image to start face detection.")
