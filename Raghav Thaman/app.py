"""
WEEK 7: STREAMLIT DEPLOYMENT APP
=================================
Scanner Identification System - Interactive Web Application

Save this file as: app.py
Run with: streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
from datetime import datetime
import json

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Scanner Identification System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        background-color: #f0f8ff;
        margin: 10px 0;
    }
    .confidence-high {
        color: #2ca02c;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .confidence-low {
        color: #d62728;
        font-weight: bold;
    }
    .metric-card {
        padding: 15px;
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_models():
    """Load trained ensemble models"""
    device = torch.device('cpu')
    
    # Define model architecture (same as training)
    def create_model(num_classes=11):
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        feature_dim = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        return model
    
    # Load models
    base_path = r"C:\msys64\home\Raghav Thaman\internship\milestone_3"
    
    models_list = []
    model_files = [
        "best_model_cpu_fast.pth",
        "finetuned_model.pth",
        "ensemble_model_2.pth",
        "ensemble_model_3.pth"
    ]
    
    for model_file in model_files:
        model_path = os.path.join(base_path, model_file)
        if os.path.exists(model_path):
            try:
                model = create_model(num_classes=11)
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                models_list.append(model)
            except:
                pass
    
    if len(models_list) == 0:
        st.error("No models found! Please check model paths.")
        return None, None
    
    # Load label encoder
    try:
        checkpoint = torch.load(os.path.join(base_path, "best_model_cpu_fast.pth"), 
                               map_location=device, weights_only=False)
        if 'label_encoder' in checkpoint:
            label_encoder = checkpoint['label_encoder']
        else:
            # Fallback to default classes
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array([
                'Canon120-1', 'Canon120-2', 'Canon220', 'Canon9000-1', 'Canon9000-2',
                'EpsonV370-1', 'EpsonV370-2', 'EpsonV39-1', 'EpsonV39-2', 'EpsonV550', 'HP'
            ])
    except:
        st.error("Could not load label encoder!")
        return None, None
    
    return models_list, label_encoder

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_prnu_residual(image, size=(160, 160)):
    """Extract PRNU residual from image"""
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = np.array(image.convert('L'))
    
    # Resize
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    
    # Denoise
    denoised = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Extract residual
    residual = image.astype(np.float32) - denoised.astype(np.float32)
    residual = residual / 127.5
    
    return residual

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_scanner(image, models, label_encoder, n_tta=5):
    """
    Predict scanner model with TTA
    """
    device = torch.device('cpu')
    
    # Extract PRNU residual
    residual = extract_prnu_residual(image)
    residual = np.expand_dims(residual, axis=-1)
    
    # Transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    input_tensor = transform(residual).unsqueeze(0).to(device)
    
    # Ensemble prediction with TTA
    all_probs = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            # Original
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs)
            
            # TTA augmentations
            for _ in range(n_tta - 1):
                # Horizontal flip
                aug_tensor = torch.flip(input_tensor, dims=[3])
                outputs = model(aug_tensor)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs)
    
    # Average all predictions
    avg_probs = torch.stack(all_probs).mean(dim=0)
    confidence, predicted_idx = torch.max(avg_probs, 1)
    
    # Get top 5 predictions
    top5_probs, top5_indices = torch.topk(avg_probs, k=min(5, len(label_encoder.classes_)))
    
    results = {
        'predicted_class': label_encoder.classes_[predicted_idx.item()],
        'confidence': confidence.item(),
        'all_probabilities': avg_probs[0].cpu().numpy(),
        'top5_classes': [label_encoder.classes_[idx] for idx in top5_indices[0].cpu().numpy()],
        'top5_confidences': top5_probs[0].cpu().numpy()
    }
    
    return results

# ============================================================
# PREDICTION LOGGING
# ============================================================
def log_prediction(image_name, prediction_result):
    """Log prediction to CSV file"""
    log_path = "prediction_logs.csv"
    
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_name': image_name,
        'predicted_scanner': prediction_result['predicted_class'],
        'confidence': f"{prediction_result['confidence']*100:.2f}%"
    }
    
    # Append to CSV
    df = pd.DataFrame([log_entry])
    if os.path.exists(log_path):
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, mode='w', header=True, index=False)
    
    return log_path

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================
def create_confidence_chart(results):
    """Create confidence chart for top predictions"""
    import plotly.graph_objects as go
    
    classes = results['top5_classes']
    confidences = results['top5_confidences'] * 100
    
    colors = ['#2ca02c' if i == 0 else '#1f77b4' for i in range(len(classes))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=confidences,
            y=classes,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{conf:.1f}%" for conf in confidences],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Top 5 Predictions",
        xaxis_title="Confidence (%)",
        yaxis_title="Scanner Model",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

# ============================================================
# MAIN APP
# ============================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Scanner Identification System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        Forensic Scanner Source Identification using Deep Learning
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    with st.spinner('Loading AI models...'):
        models, label_encoder = load_models()
    
    if models is None:
        st.error("Failed to load models. Please check model files.")
        return
    
    st.success(f"✓ Loaded {len(models)} model(s) successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.markdown("---")
        st.subheader("Model Information")
        st.info(f"""
        **Architecture:** ResNet18 Ensemble  
        **Number of Models:** {len(models)}  
        **Scanner Classes:** {len(label_encoder.classes_)}  
        **Accuracy:** 92.26% (Validation)
        """)
        
        st.markdown("---")
        st.subheader("About")
        st.markdown("""
        This system identifies the source scanner 
        used to scan a document by analyzing unique 
        scanner fingerprints (PRNU patterns).
        
        **Use Cases:**
        - Digital Forensics
        - Document Authentication
        - Legal Evidence Verification
        """)
        
        st.markdown("---")
        st.subheader("Supported Scanners")
        with st.expander("View all scanners"):
            for scanner in sorted(label_encoder.classes_):
                st.write(f"• {scanner}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Image</h2>', 
                   unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a scanned image...",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'],
            help="Upload a scanned document image for scanner identification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.markdown("**Image Details:**")
            st.write(f"• Filename: `{uploaded_file.name}`")
            st.write(f"• Size: `{image.size[0]} x {image.size[1]} pixels`")
            st.write(f"• Format: `{image.format}`")
    
    with col2:
        st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', 
                   unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Predict button
            if st.button("🔍 Identify Scanner", type="primary", use_container_width=True):
                with st.spinner('Analyzing scanner fingerprint...'):
                    # Make prediction
                    results = predict_scanner(image, models, label_encoder, n_tta=5)
                    
                    # Log prediction
                    log_path = log_prediction(uploaded_file.name, results)
                
                # Display results
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                # Main prediction
                confidence = results['confidence'] * 100
                
                if confidence >= 80:
                    conf_class = "confidence-high"
                    emoji = "✅"
                elif confidence >= 60:
                    conf_class = "confidence-medium"
                    emoji = "⚠️"
                else:
                    conf_class = "confidence-low"
                    emoji = "❌"
                
                st.markdown(f"""
                ### {emoji} Predicted Scanner
                <div style='font-size: 2rem; font-weight: bold; color: #1f77b4;'>
                    {results['predicted_class']}
                </div>
                <div style='font-size: 1.5rem;' class='{conf_class}'>
                    Confidence: {confidence:.2f}%
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Confidence chart
                st.plotly_chart(create_confidence_chart(results), 
                              use_container_width=True)
                
                # Detailed probabilities
                with st.expander("📊 View All Class Probabilities"):
                    prob_df = pd.DataFrame({
                        'Scanner Model': label_encoder.classes_,
                        'Probability (%)': results['all_probabilities'] * 100
                    }).sort_values('Probability (%)', ascending=False)
                    
                    st.dataframe(
                        prob_df.style.background_gradient(
                            subset=['Probability (%)'], 
                            cmap='Greens'
                        ).format({'Probability (%)': '{:.2f}%'}),
                        use_container_width=True,
                        height=400
                    )
                
                # Download results
                st.markdown("---")
                st.markdown("### 💾 Download Results")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    # JSON download
                    result_json = {
                        'image': uploaded_file.name,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'prediction': results['predicted_class'],
                        'confidence': f"{confidence:.2f}%",
                        'top5_predictions': {
                            cls: f"{conf*100:.2f}%" 
                            for cls, conf in zip(results['top5_classes'], 
                                               results['top5_confidences'])
                        }
                    }
                    
                    st.download_button(
                        label="📄 Download JSON",
                        data=json.dumps(result_json, indent=2),
                        file_name=f"prediction_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col_b:
                    # CSV download
                    if os.path.exists(log_path):
                        with open(log_path, 'r') as f:
                            st.download_button(
                                label="📊 Download Logs CSV",
                                data=f.read(),
                                file_name="prediction_logs.csv",
                                mime="text/csv"
                            )
        else:
            st.info("👆 Upload an image to begin scanner identification")
    
    # Statistics section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📈 Prediction Statistics</h2>', 
               unsafe_allow_html=True)
    
    if os.path.exists("prediction_logs.csv"):
        logs_df = pd.read_csv("prediction_logs.csv")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{len(logs_df)}</h3>
                <p>Total Predictions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            unique_scanners = logs_df['predicted_scanner'].nunique()
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{unique_scanners}</h3>
                <p>Unique Scanners</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_conf = logs_df['confidence'].str.rstrip('%').astype(float).mean()
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{avg_conf:.1f}%</h3>
                <p>Avg Confidence</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            recent = logs_df['timestamp'].iloc[-1] if len(logs_df) > 0 else "N/A"
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='font-size: 1rem;'>{recent}</h3>
                <p>Last Prediction</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent predictions table
        st.markdown("### 📝 Recent Predictions")
        st.dataframe(
            logs_df.tail(10).sort_values('timestamp', ascending=False),
            use_container_width=True
        )
    else:
        st.info("No predictions logged yet. Upload an image to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        Scanner Identification System | Week 7 Deployment | Powered by PyTorch & Streamlit
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# RUN APP
# ============================================================
if __name__ == "__main__":
    main()
