import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import soundfile as sf
import io

# Page configuration
st.set_page_config(
    page_title="COPD Detection System",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS with New Color Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Color Theme Variables:
       Navy Blue: #2D3960 - Primary headers and buttons
       Coral Pink: #F0907C - Accent borders and highlights
       Light Beige: #F0E2D5 - Background and subtle elements
       Warm Yellow: #F6C370 - Secondary accents and status values
       
       To change colors, replace these hex codes throughout the CSS:
       - #2D3960 (Navy Blue) - for primary elements
       - #F0907C (Coral Pink) - for accents
       - #F0E2D5 (Light Beige) - for backgrounds
       - #F6C370 (Warm Yellow) - for highlights
    */
    
    .main {
        background: linear-gradient(135deg, #F0E2D5 0%, #ffffff 50%, #F0E2D5 100%);
        padding: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #F0E2D5 0%, #ffffff 50%, #F0E2D5 100%);
    }
    
    /* Header */
    .header-container {
        background: linear-gradient(135deg, #2D3960 0%, #3d4a75 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(45, 57, 96, 0.2);
    }
    
    .main-title {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.15);
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: #F0E2D5;
        font-size: 1.2rem;
        margin-top: 0.8rem;
        font-weight: 400;
    }
    
    /* Cards */
    .info-card {
        background: white;
        padding: 1.8rem;
        border-radius: 16px;
        border-left: 5px solid #F6C370;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    .success-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8f4 100%);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        border-left: 5px solid #4caf50;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.15);
    }
    
    .success-title {
        color: #1b5e20;
        font-weight: 700;
        font-size: 1rem;
        margin: 0 0 0.4rem 0;
    }
    
    .success-text {
        color: #2e7d32;
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Upload Section */
    .upload-section {
        background: white;
        padding: 3rem;
        border-radius: 16px;
        margin: 2rem 0;
        text-align: center;
        border: 3px dashed #F0907C;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #2D3960;
        background: #FFF9F5;
        transform: translateY(-2px);
    }
    
    /* Results */
    .result-positive {
        background: linear-gradient(135deg, #ffebee 0%, #fff5f5 100%);
        padding: 3rem;
        border-radius: 16px;
        margin: 2rem 0;
        border-left: 8px solid #d32f2f;
        box-shadow: 0 6px 20px rgba(211, 47, 47, 0.2);
    }
    
    .result-negative {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8f4 100%);
        padding: 3rem;
        border-radius: 16px;
        margin: 2rem 0;
        border-left: 8px solid #388e3c;
        box-shadow: 0 6px 20px rgba(56, 142, 60, 0.2);
    }
    
    .result-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 1rem;
        color: #1f2937;
    }
    
    .result-confidence {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 1.5rem 0;
        background: linear-gradient(135deg, #2D3960 0%, #F6C370 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .result-description {
        font-size: 1.1rem;
        line-height: 1.9;
        color: #4b5563;
    }
    
    /* Disclaimer */
    .disclaimer-box {
        background: linear-gradient(135deg, #fff3e0 0%, #fff8e1 100%);
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        color: #e65100;
        border-left: 5px solid #F6C370;
        box-shadow: 0 4px 12px rgba(246, 195, 112, 0.15);
    }
    
    .disclaimer-title {
        font-weight: 800;
        font-size: 1.2rem;
        margin-bottom: 0.8rem;
        color: #2D3960;
    }
    
    /* Sidebar */
    .sidebar-header {
        background: linear-gradient(135deg, #2D3960 0%, #3d4a75 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        font-weight: 800;
        margin-bottom: 1.5rem;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(45, 57, 96, 0.25);
    }
    
    .sidebar-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #F0E2D5;
    }
    
    .status-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background: linear-gradient(135deg, #FFF9F5 0%, #ffffff 100%);
        border-radius: 10px;
        margin: 0.8rem 0;
        border-left: 4px solid #F6C370;
    }
    
    .status-label {
        color: #6b7280;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .status-value {
        color: #2D3960;
        font-weight: 800;
        font-size: 1rem;
    }
    
    /* Button */
    .stButton>button {
        background: linear-gradient(135deg, #2D3960 0%, #F6C370 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 1.2rem 3rem;
        border: none;
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(45, 57, 96, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1f2740 0%, #f0a850 100%);
        box-shadow: 0 8px 24px rgba(45, 57, 96, 0.4);
        transform: translateY(-3px);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    /* Section Headers */
    .section-header {
        color: #2D3960;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.8rem;
        border-bottom: 4px solid #F0907C;
    }
    
    /* Intro Box */
    .intro-box {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        border: 2px solid #F0E2D5;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    
    .intro-icon {
        color: #F0907C;
        font-size: 1.8rem;
        margin-right: 0.8rem;
    }
    
    /* Model Info Box */
    .model-info-box {
        background: linear-gradient(135deg, #2D3960 0%, #3d4a75 100%);
        padding: 1.8rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 6px 20px rgba(45, 57, 96, 0.35);
    }
    
    .model-info-item {
        display: flex;
        justify-content: space-between;
        padding: 0.8rem 0;
        border-bottom: 1px solid rgba(240, 226, 213, 0.3);
    }
    
    .model-info-item:last-child {
        border-bottom: none;
    }
    
    .model-label {
        color: #F0E2D5;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .model-value {
        font-weight: 700;
        font-size: 1rem;
        color: #F6C370;
    }
    
    /* Numbered List */
    .numbered-list {
        counter-reset: item;
        list-style: none;
        padding: 0;
    }
    
    .numbered-list li {
        counter-increment: item;
        padding: 1rem 0 1rem 3rem;
        position: relative;
        color: #4b5563;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    .numbered-list li::before {
        content: counter(item);
        position: absolute;
        left: 0;
        background: linear-gradient(135deg, #2D3960 0%, #F6C370 100%);
        color: white;
        width: 2rem;
        height: 2rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(45, 57, 96, 0.3);
    }
    
    /* Bullet List */
    .bullet-list {
        list-style: none;
        padding: 0;
    }
    
    .bullet-list li {
        padding: 0.7rem 0;
        padding-left: 2rem;
        position: relative;
        color: #4b5563;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    .bullet-list li::before {
        content: "●";
        position: absolute;
        left: 0.5rem;
        color: #F0907C;
        font-size: 1.2rem;
    }
    
    /* Audio Player */
    audio {
        width: 100%;
        margin: 1rem 0;
        border-radius: 10px;
    }
    
    /* File Uploader */
    .stFileUploader {
        background: transparent;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F0E2D5;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #2D3960;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #F0907C;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('copd_mfcc_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Extract MFCC features from audio
def extract_mfcc_features(audio_file, target_shape, sr=22050):
    try:
        target_time = target_shape[1] if target_shape[1] is not None else 862
        target_features = target_shape[2] if target_shape[2] is not None else 40
        
        y, sr = librosa.load(audio_file, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=target_features)
        mfcc = mfcc.T
        
        if mfcc.shape[0] < target_time:
            pad_width = target_time - mfcc.shape[0]
            mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:target_time, :]
        
        if mfcc.shape[1] < target_features:
            pad_width = target_features - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        elif mfcc.shape[1] > target_features:
            mfcc = mfcc[:, :target_features]
        
        return mfcc
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Predict COPD
def predict_copd(model, mfcc_features):
    try:
        mfcc_features = mfcc_features.reshape(1, mfcc_features.shape[0], mfcc_features.shape[1], 1)
        prediction = model.predict(mfcc_features, verbose=0)
        probability = prediction[0][0]
        return probability
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Main UI
def main():
    # Header
    st.markdown("""
        <div class="header-container">
            <h1 class="main-title"> COPD DETECTION SYSTEM</h1>
            <p class="subtitle">Advanced Audio Analysis for Respiratory Health Assessment</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create columns
    col_main, col_sidebar = st.columns([2, 1])
    
    with col_sidebar:
        # How to Use
        st.markdown('<div class="sidebar-header">How to Use</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="sidebar-card">
                <ol class="numbered-list">
                    <li>Upload an audio file (WAV, MP3, FLAC)</li>
                    <li>Click the Analyze Audio button</li>
                    <li>Review the detection results</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        
        # System Status
        st.markdown('<div class="sidebar-header">System Status</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="sidebar-card">
                <div class="status-item">
                    <span class="status-label">Model Status</span>
                    <span class="status-value">Active</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Detection Threshold</span>
                    <span class="status-value">85%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Supported Formats
        st.markdown('<div class="sidebar-header">Supported Formats</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="sidebar-card">
                <ul class="bullet-list">
                    <li>WAV - Waveform Audio</li>
                    <li>MP3 - MPEG Audio</li>
                    <li>FLAC - Lossless Audio</li>
                    <li>OGG - Ogg Vorbis</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # Load model and display info
        model = load_model()
        if model:
            st.markdown('<div class="sidebar-header">Model Information</div>', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="model-info-box">
                    <div class="model-info-item">
                        <span class="model-label">Input Shape:</span>
                        <span class="model-value">{model.input_shape}</span>
                    </div>
                    <div class="model-info-item">
                        <span class="model-label">Output Shape:</span>
                        <span class="model-value">{model.output_shape}</span>
                    </div>
                    <div class="model-info-item">
                        <span class="model-label">Model Type:</span>
                        <span class="model-value">CNN + LSTM</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    with col_main:
        # Success card
        if model:
            st.markdown("""
                <div class="success-card">
                    <p class="success-title"> MODEL STATUS</p>
                    <p class="success-text">Model loaded successfully and ready for analysis</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Introduction
        st.markdown("""
            <div class="intro-box">
                <p style="margin: 0; color: #4b5563; font-size: 1.05rem; line-height: 1.8;">
                    <span class="intro-icon"></span>
                    This system analyzes audio recordings including breathing sounds, cough patterns, 
                    and speech to detect potential Chronic Obstructive Pulmonary Disease (COPD) indicators 
                    using advanced machine learning algorithms.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Upload section
        st.markdown('<p class="section-header"> Upload Audio File</p>', unsafe_allow_html=True)
        
        audio_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'ogg'],
            help="Upload a breathing sound, cough, or speech recording",
            label_visibility="collapsed"
        )
        
        if audio_file is not None:
            st.markdown('<p class="section-header"> Audio Preview</p>', unsafe_allow_html=True)
            st.audio(audio_file, format='audio/wav')
            
            # Analyze button
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_button = st.button(" ANALYZE AUDIO", use_container_width=True)
            
            if analyze_button:
                if model is None:
                    st.error("Model could not be loaded. Please ensure 'copd_mfcc_model.keras' is in the same directory.")
                else:
                    with st.spinner(' Analyzing audio... Please wait.'):
                        # Extract features
                        mfcc_features = extract_mfcc_features(audio_file, model.input_shape)
                        
                        if mfcc_features is not None:
                            # Make prediction
                            probability = predict_copd(model, mfcc_features)
                            
                            if probability is not None:
                                # Display results
                                st.markdown('<p class="section-header"> Analysis Results</p>', unsafe_allow_html=True)
                                
                                # COPD Detection Logic: 85% threshold
                                is_copd = probability >= 0.85
                                
                                if is_copd:
                                    st.markdown(f"""
                                    <div class="result-positive">
                                        <div class="result-title"> COPD DETECTED</div>
                                        <div class="result-confidence">{probability*100:.2f}%</div>
                                        <div class="result-description">
                                            The audio analysis has detected COPD indicators with high confidence (≥85%). 
                                            This result indicates significant abnormal respiratory patterns that warrant immediate medical evaluation. 
                                            Please consult a healthcare professional as soon as possible for proper diagnosis and treatment.
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="result-negative">
                                        <div class="result-title"> NO COPD DETECTED</div>
                                        <div class="result-confidence">{probability*100:.2f}%</div>
                                        <div class="result-description">
                                            The audio analysis does not show COPD indicators with sufficient confidence (below 85% threshold). 
                                            The respiratory patterns appear to be within acceptable ranges or show insufficient evidence for COPD detection. 
                                            However, if you have concerns about your respiratory health, please consult a healthcare professional.
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("""
            <div class="disclaimer-box">
                <div class="disclaimer-title"> Medical Disclaimer</div>
                <p style="margin: 0; line-height: 1.8; font-size: 1.05rem;">
                    This system is for screening purposes only and should not replace professional 
                    medical diagnosis. Always consult with a qualified healthcare provider for 
                    proper diagnosis and treatment.
                </p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()