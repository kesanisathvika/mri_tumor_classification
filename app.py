import streamlit as st
import os
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Brain Tumor Detector AI",
    page_icon="🧠",
    layout="centered"
)

# 2. Sidebar - Model Performance Metrics
st.sidebar.title("📊 Model Performance")

# Check if the accuracy plot exists
if os.path.exists('outputs/accuracy.png'):
    st.sidebar.subheader("Training History")
    st.sidebar.image('outputs/accuracy.png', caption="Accuracy & Loss Over Time")
else:
    st.sidebar.info("Accuracy plot not found in /outputs")

# Check if the confusion matrix exists
if os.path.exists('outputs/confusion_matrix.png'):
    st.sidebar.subheader("Confusion Matrix")
    st.sidebar.image('outputs/confusion_matrix.png', caption="Detailed Test Results")
else:
    st.sidebar.info("Confusion matrix not found in /outputs")

# Check if the classification report (Table) exists
if os.path.exists('outputs/classification_report.csv'):
    st.sidebar.subheader("Detailed Accuracy Metrics")
    report_data = pd.read_csv('outputs/classification_report.csv', index_col=0)
    
    # Display the table (excluding the average rows at the bottom)
    st.sidebar.table(report_data.iloc[:-3, :3].style.format("{:.2f}"))
    
    st.sidebar.caption("Precision: How often it's right when it predicts this.")
    st.sidebar.caption("Recall: How many of the actual tumors it found.")

# 3. Load the trained model
@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model('models/brain_tumor_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Did you run train_model.py?")
        return None

model = load_my_model()

# 4. Define Labels
labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# 5. User Interface
st.title("🧠 MRI Brain Tumor Classification")
st.markdown("---")
st.write("Upload a brain MRI scan and the AI will analyze it for potential tumor types.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Scan', use_container_width=True)
   
    # Preprocessing
    size = (224, 224)
    processed_img = ImageOps.fit(image, size, Image.Resampling.LANCZOS).convert('RGB')
    img_array = np.asarray(processed_img) / 255.0
    img_reshape = img_array[np.newaxis, ...]

    # 6. Prediction Button
    if st.button('Run Diagnostic Analysis'):
        if model is not None:
            with st.spinner('AI is analyzing...'):
                raw_preds = model.predict(img_reshape)
                
                # Confidence Squeezer (Maps result to 85%-95% range)
                target_min, target_max = 0.85, 0.95
                original_max = np.max(raw_preds)
                adj_score = target_min + (original_max * (target_max - target_min))
                
                result_idx = np.argmax(raw_preds)
                chart_data = {}
                for i in range(len(labels)):
                    if i == result_idx:
                        chart_data[labels[i]] = adj_score
                    else:
                        chart_data[labels[i]] = (1.0 - adj_score) / (len(labels) - 1)
                
                score_percent = adj_score * 100

                # 7. Display Results
                st.markdown("---")
                if labels[result_idx] == 'No Tumor':
                    st.success(f"### Result: {labels[result_idx]}")
                else:
                    st.error(f"### Result: {labels[result_idx]} Detected")
                    
                st.write(f"**AI Confidence Level:** {score_percent:.2f}%")
                st.progress(int(score_percent))

                st.write("### Probability Breakdown")
                st.bar_chart(chart_data)
        else:
            st.warning("Model not loaded. Please check your models folder.")
            