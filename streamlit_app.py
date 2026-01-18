"""
BLIP-2 Emotion Recognition Dashboard
Streamlit app for emotion and Action Unit analysis using fine-tuned BLIP-2 model
"""

import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel
import tempfile

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Emotion Recognition Dashboard",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.2em;
        font-weight: 600;
    }
    .emotion-card {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-card {
        padding: 15px;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-card {
        padding: 15px;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# FACE DETECTION & PREPROCESSING
# ============================================================

class FaceAlignmentPreprocessor:
    """Face detection, alignment, and cropping using OpenCV"""
    
    def __init__(self, output_size=(336, 336)):
        self.output_size = output_size
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def detect_faces(self, image):
        """Detect faces in image and return face regions"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    
    def align_and_crop(self, image):
        """Align face using eye detection and crop to target size"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, "No faces detected in the image"
        
        # Take the largest face found
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        
        # Detect eyes within the face ROI
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate to identify left vs right
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye_center = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
            right_eye_center = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
            
            # Calculate angle for alignment
            dY = right_eye_center[1] - left_eye_center[1]
            dX = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Rotate
            center = (int(w / 2), int(h / 2))
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(roi_color, M, (w, h))
            aligned_face = cv2.resize(rotated, self.output_size)
        else:
            # Fallback: Just crop the face box if eyes aren't found
            aligned_face = cv2.resize(roi_color, self.output_size)
        
        return aligned_face, "Face detected and aligned successfully"


# ============================================================
# MODEL LOADING (CACHED)
# ============================================================

@st.cache_resource
def load_model(model_path):
    """Load fine-tuned BLIP-2 model with LoRA adapters"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load base model
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load fine-tuned LoRA adapters
        if os.path.exists(model_path):
            model = PeftModel.from_pretrained(model, model_path)
            return model, processor, device, True
        else:
            return model, processor, device, False
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, False


# ============================================================
# EMOTION ANALYSIS
# ============================================================

def analyze_emotion(model, processor, image, device):
    """Generate emotion analysis using fine-tuned BLIP-2"""
    
    prompt = (
        "Analyze this facial image and identify:\n"
        "1. Which emotions are present (Surprise, Fear, Disgust, Happiness, Sadness, Anger)\n"
        "2. The facial Action Units (AUs) involved\n"
        "Please explain the connection between the AUs and the emotions."
    )
    
    # Prepare inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    
    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1>😊 Facial Emotion Recognition Dashboard</h1>
            <p style='font-size: 1.1em; color: #666;'>
                Fine-tuned BLIP-2 Model with Action Unit Analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Configuration")
    
    model_path = st.sidebar.text_input(
        "Model Path (LoRA adapters)",
        value="./blip2-emotion-rafce-final",
        help="Path to the fine-tuned LoRA adapters directory"
    )
    
    st.sidebar.markdown("---")
    
    # Model info
    st.sidebar.markdown("### 📊 Model Info")
    st.sidebar.info(
        """
        **Model**: BLIP-2 OPT 2.7B + LoRA  
        **Emotions**: 6 classes  
        - Surprise, Fear, Disgust  
        - Happiness, Sadness, Anger  
        **Features**: Multi-label emotion detection with Action Unit analysis
        """
    )
    
    # Check device
    device_info = "🟢 GPU Available" if torch.cuda.is_available() else "🔴 CPU Mode"
    st.sidebar.write(device_info)
    
    # Load model
    st.sidebar.markdown("---")
    with st.sidebar.status("Loading model...", expanded=True) as status:
        model, processor, device, lora_loaded = load_model(model_path)
        
        if model is not None:
            status.update(
                label="✅ Model loaded successfully!",
                state="complete"
            )
        else:
            status.update(
                label="❌ Failed to load model",
                state="error"
            )
    
    if model is None:
        st.error("""
        ❌ **Model Loading Failed**
        
        Please ensure:
        1. LoRA adapters are in the correct directory
        2. Base model (Salesforce/blip2-opt-2.7b) can be downloaded
        3. You have sufficient GPU/CPU memory
        """)
        return
    
    if not lora_loaded:
        st.warning("""
        ⚠️ **LoRA Adapters Not Found**
        
        Using base model only. Fine-tuned parameters not loaded.
        Please verify the model path.
        """)
    
    # Initialize preprocessor
    preprocessor = FaceAlignmentPreprocessor()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎥 Upload & Analyze", "📚 Batch Processing", "ℹ️ About"])
    
    # ============================================================
    # TAB 1: UPLOAD & ANALYZE
    # ============================================================
    with tab1:
        st.markdown("### 📤 Upload an Image")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Upload a facial image for emotion analysis"
            )
        
        with col2:
            analyze_button = st.button("🔍 Analyze Emotion", use_container_width=True)
        
        if uploaded_file is not None:
            # Display uploaded image
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Original Image")
                original_image = Image.open(uploaded_file)
                st.image(original_image, use_column_width=True)
            
            # Convert to OpenCV format
            image_array = np.array(original_image)
            if len(image_array.shape) == 2:  # Grayscale
                image_cv = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            elif image_array.shape[2] == 4:  # RGBA
                image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Face detection and alignment
            with col2:
                st.markdown("#### Processed Face")
                
                with st.spinner("🔄 Processing face..."):
                    aligned_face, message = preprocessor.align_and_crop(image_cv)
                
                if aligned_face is not None:
                    # Convert back to RGB for display
                    face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                    st.image(face_rgb, use_column_width=True)
                    
                    # Success message
                    st.markdown(f"""
                    <div class='success-card'>
                    ✅ {message}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='error-card'>
                    ❌ {message}
                    </div>
                    """, unsafe_allow_html=True)
                    aligned_face = image_cv
            
            # Emotion analysis
            if analyze_button and aligned_face is not None:
                with st.spinner("🤖 Analyzing emotions..."):
                    try:
                        # Convert to PIL for model
                        face_pil = Image.fromarray(
                            cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                        )
                        
                        # Get prediction
                        prediction = analyze_emotion(model, processor, face_pil, device)
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### 🎭 Emotion Analysis Results")
                        
                        st.markdown(f"""
                        <div class='emotion-card'>
                        <h3>Model Prediction</h3>
                        <p>{prediction}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Copy button
                        col1, col2 = st.columns([0.8, 0.2]
                        )
                        with col2:
                            st.button(
                                "📋 Copy Result",
                                on_click=lambda: st.write(prediction)
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
    
    # ============================================================
    # TAB 2: BATCH PROCESSING
    # ============================================================
    with tab2:
        st.markdown("### 📁 Batch Processing")
        st.info(
            "Upload multiple images to process them in batch. "
            "Results will be saved to a CSV file."
        )
        
        uploaded_files = st.file_uploader(
            "Choose multiple images",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            help="Select multiple facial images for batch analysis"
        )
        
        if len(uploaded_files) > 0:
            if st.button("🚀 Process Batch", use_container_width=True):
                progress_bar = st.progress(0)
                results_container = st.container()
                
                results = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    with results_container.status(
                        f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}",
                        expanded=False
                    ):
                        try:
                            # Load image
                            original_image = Image.open(uploaded_file)
                            image_array = np.array(original_image)
                            
                            if len(image_array.shape) == 2:
                                image_cv = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
                            elif image_array.shape[2] == 4:
                                image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
                            else:
                                image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                            
                            # Process face
                            aligned_face, message = preprocessor.align_and_crop(image_cv)
                            
                            if aligned_face is not None:
                                face_pil = Image.fromarray(
                                    cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                                )
                                prediction = analyze_emotion(
                                    model, processor, face_pil, device
                                )
                                
                                results.append({
                                    "Filename": uploaded_file.name,
                                    "Status": "✅ Success",
                                    "Prediction": prediction[:100] + "..." if len(prediction) > 100 else prediction
                                })
                                st.write("✅ Face detected and emotion analyzed")
                            else:
                                results.append({
                                    "Filename": uploaded_file.name,
                                    "Status": "❌ No face detected",
                                    "Prediction": "N/A"
                                })
                                st.write("❌ No face detected")
                        
                        except Exception as e:
                            results.append({
                                "Filename": uploaded_file.name,
                                "Status": f"❌ Error: {str(e)[:50]}",
                                "Prediction": "N/A"
                            })
                            st.write(f"❌ Error: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                # Display results table
                st.markdown("---")
                st.markdown("### 📊 Batch Results")
                
                import pandas as pd
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="emotion_analysis_results.csv",
                    mime="text/csv"
                )
    
    # ============================================================
    # TAB 3: ABOUT
    # ============================================================
    with tab3:
        st.markdown("""
        ## 📖 About This Dashboard
        
        This Streamlit application provides an interactive interface for emotion recognition
        using a fine-tuned **BLIP-2** (Bootstrapping Language-Image Pre-training) model.
        
        ### 🎯 Features
        
        - **Face Detection**: Automatic detection and alignment of faces in images
        - **Emotion Recognition**: Identify 6 emotions (Surprise, Fear, Disgust, Happiness, Sadness, Anger)
        - **Action Unit Analysis**: Extract facial Action Units linked to emotions
        - **Multi-label Support**: Recognize multiple emotions in a single face
        - **Batch Processing**: Process multiple images efficiently
        
        ### 🏗️ Architecture
        
        **Base Model**: Salesforce/blip2-opt-2.7b (2.7B parameters)  
        **Fine-tuning**: LoRA (Low-Rank Adaptation) for efficient adaptation  
        **Face Preprocessing**: OpenCV Haar Cascade + Eye Detection + Alignment  
        **Input Resolution**: 336×336 pixels (standard for BLIP-2)
        
        ### 📊 Model Performance
        
        - **Training Epochs**: 3
        - **Batch Size**: 4 (effective: 16 with gradient accumulation)
        - **Learning Rate**: 2e-4 with linear warmup
        - **Optimizer**: AdamW with weight decay
        - **Mixed Precision**: FP16 enabled
        
        ### 🎓 Emotions Supported
        
        | Emotion | Description |
        |---------|-------------|
        | 😮 Surprise | Sudden unexpected expression |
        | 😨 Fear | Expression of anxiety or dread |
        | 😖 Disgust | Expression of dislike or revulsion |
        | 😊 Happiness | Expression of joy and contentment |
        | 😢 Sadness | Expression of sorrow or grief |
        | 😠 Anger | Expression of displeasure or rage |
        
        ### 💡 Tips for Best Results
        
        1. **Image Quality**: Use clear, well-lit facial images
        2. **Face Size**: Ensure the face occupies a reasonable portion of the image
        3. **Face Position**: Front-facing faces work best (30-45° angle acceptable)
        4. **Single Face**: Process one face per image for best accuracy
        
        ### 📝 Output Format
        
        The model generates natural language descriptions including:
        - Identified emotions
        - Associated facial Action Units (AUs)
        - Explanations of AU-emotion connections
        
        ### ⚙️ Technical Details
        
        - **Framework**: PyTorch + Transformers
        - **Preprocessing**: OpenCV
        - **UI**: Streamlit
        - **Device**: GPU/CPU (auto-detected)
        
        ### 📚 References
        
        - [BLIP-2 Paper](https://arxiv.org/abs/2301.12597)
        - [LoRA Paper](https://arxiv.org/abs/2106.09685)
        - [Facial Action Units](https://en.wikipedia.org/wiki/Facial_action_coding_system)
        
        ---
        
        **Version**: 1.0  
        **Last Updated**: January 2026
        """)


if __name__ == "__main__":
    main()
