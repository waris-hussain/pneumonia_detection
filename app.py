import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="centered"
)

# --- Load Model ---
@st.cache_resource
def load_learner():
    try:
        # Load the model we saved in train.py
        model = tf.keras.models.load_model('model.h5', compile=False)
        return model
    except Exception as e:
        st.error("Model file 'model.h5' not found.")
        st.info("Please run 'python train.py' first to generate the model.")
        return None

model = load_learner()

# --- UI Design ---
st.title("🫁 Pneumonia Detection System")
st.markdown("""
<style>
    .stAlert { margin-top: 20px; }
    .reportview-container { background: #f0f2f6 }
</style>
""", unsafe_allow_html=True)

st.write("**AI-Powered Diagnosis Assistance**")
st.info("Upload a chest X-Ray image (JPEG/PNG) to analyze for signs of pneumonia.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload X-Ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Show user the image
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption='Uploaded Scan', use_container_width=True)
    
    # --- Prediction Logic ---
    if st.button("Analyze Image", type="primary"):
        if model is None:
            st.error("Model is not loaded. Cannot predict.")
        else:
            with st.spinner("Analyzing image patterns..."):
                # 1. Preprocess the image to match model input
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                img_array = np.asarray(image)
                
                # Handle Greyscale (2D) or RGBA (4 channels)
                if img_array.ndim == 2:
                    img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[2] == 4:
                     img_array = img_array[:,:,:3]
                     
                # Normalize pixels to [0, 1]
                normalized_image_array = (img_array.astype(np.float32) / 255.0)
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                data[0] = normalized_image_array

                # 2. Make Prediction
                prediction = model.predict(data)
                
                # Classes are sorted alphabetically by ImageDataGenerator: Normal, Pneumonia
                classes = ['Normal', 'Pneumonia']
                class_idx = np.argmax(prediction[0])
                confidence = prediction[0][class_idx] * 100
                result = classes[class_idx]
                
                # 3. Display Results
                with col2:
                    st.subheader("Analysis Result")
                    if result == 'Pneumonia':
                        st.error(f"⚠️ **Pneumonia Detected**")
                        st.write("The model identified patterns consistent with pneumonia.")
                    else:
                        st.success(f"✅ **Normal**")
                        st.write("Calculate patterns appear healthy.")
                        
                    st.metric("Confidence Level", f"{confidence:.2f}%")
                    st.progress(int(confidence))

                # Optional: Probability Breakdown
                st.write("---")
                st.caption("Probability Distribution")
                chart_data = {"Normal": prediction[0][0], "Pneumonia": prediction[0][1]}
                st.bar_chart(chart_data)

st.markdown("---")
st.caption("Note: This project is for educational purposes only.")
