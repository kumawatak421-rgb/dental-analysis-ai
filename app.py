import gdown
import os

url = 'https://drive.google.com/file/d/1WF9iDdhw_J0O1A1gXgxrfsWtHhTVjD0x/view?usp=sharing'
output = 'dental_disease_model.h5'

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)
import streamlit as st
import tflite_runtime.interpreter as tflite

import cv2
import numpy as np
import os

# Page Configuration
st.set_page_config(page_title="DENTAL ANALYSIS AI", page_icon="🦷", layout="centered")

st.title("🦷 DENTAL ANALYSIS AI")
st.write("Upload a dental X-ray image (periapical or OPG) to analyze conditions like bone loss or abscess instantly.")
st.markdown("### **Developed & Designed by: DR. AJAY KUMAWAT**")
# 1. Load Trained Model
@st.cache_resource
def load_ai_model():
    model_path = 'dental_disease_model.h5'
    if os.path.exists(model_path):
       return load_model(model_path)
    return None

model = load_ai_model()

if model is None:
    st.error("⚠️ Error: 'dental_disease_model.h5' model file nahi mili. Pehle model ko train karke save karein!")
else:
    # Classes jo training ke waqt thi
    class_names = ['normal_periapical', 'normal_opg', 'periapical_abscess', 'periapical_bone_loss', 'periapical_bone_loss_opg']

    # 2. File Uploader for User
    uploaded_file = st.file_uploader("Choose a Dental X-Ray Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1) # type: ignore
        
        # Safe fallback if imdecode returns None
        if opencv_image is None:
            # Re-read properly using standard cv2 flags
            opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded X-Ray")
            # Convert BGR to RGB for correct display in Streamlit
            st.image(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("AI Analysis Status")
            with st.spinner("Processing image with CLAHE & analyzing..."):
                
                # 3. Apply CLAHE Preprocessing (Jaise training me kiya tha)
                img_resized = cv2.resize(opencv_image, (224, 224))
                gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl_img = clahe.apply(gray)
                
                img_final = cv2.cvtColor(cl_img, cv2.COLOR_GRAY2RGB)
                
                # Normalization & Reshaping
                img_array = img_final.astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # 4. Predict
                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                
                predicted_class_index = np.argmax(predictions[0])
                confidence = 100 * np.max(score)
                result_class = class_names[predicted_class_index]

            # Display Results nicely
            st.success("Analysis Complete!")
            st.markdown(f"**Condition:** `{result_class}`")
            st.markdown(f"**Confidence:** `{confidence:.2f}%`")

            # Progress bar for visual confidence
            st.progress(float(confidence / 100.0))