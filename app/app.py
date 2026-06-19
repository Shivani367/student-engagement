import os
import sys
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path so we can import models and utils
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.extract_features import extract_features
from models.stgcn import STGCNClassifier
from models.bigru import BiGRUClassifier
from models.convnext1d import ConvNeXt1DClassifier
from ensemble import EnsembleClassifier

# Set page config for beautiful layout
st.set_page_config(
    page_title="Student Engagement Detector",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics (dark mode, glassmorphism, gradients, modern fonts)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #020617 100%);
        color: #f8fafc;
    }
    
    .main-title {
        background: linear-gradient(90deg, #38bdf8 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
        padding-top: 10px;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Glassmorphic cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.25);
    }
    
    .metric-title {
        color: #cbd5e1;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    /* Sleek styling for prediction display cards */
    .prediction-card {
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
        font-weight: 600;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    
    .class-0 {
        background: linear-gradient(135deg, #ef4444 0%, #7f1d1d 100%);
        border: 1px solid #f87171;
        color: #fee2e2;
    }
    
    .class-1 {
        background: linear-gradient(135deg, #f97316 0%, #7c2d12 100%);
        border: 1px solid #fb923c;
        color: #ffedd5;
    }
    
    .class-2 {
        background: linear-gradient(135deg, #10b981 0%, #064e3b 100%);
        border: 1px solid #34d399;
        color: #ecfdf5;
    }
    
    .class-3 {
        background: linear-gradient(135deg, #6366f1 0%, #312e81 100%);
        border: 1px solid #818cf8;
        color: #e0e7ff;
    }
    
    .pulse {
        animation: pulse-animation 2s infinite;
    }
    
    @keyframes pulse-animation {
        0% {
            box-shadow: 0 0 0 0 rgba(129, 140, 248, 0.4);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(129, 140, 248, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(129, 140, 248, 0);
        }
    }
</style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown('<div class="main-title">AI Student Engagement Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predicting student attention level in online learning videos using Deep Learning & Ensemble voting</div>', unsafe_allow_html=True)

# Class Map and color themes
CLASS_MAP = {
    0: "0: Disengaged (Needs Intervention)",
    1: "1: Low Engaged (Slightly Distracted)",
    2: "2: Engaged (Actively Learning)",
    3: "3: Highly Engaged (Peak Interaction)"
}
CLASS_STYLES = {
    0: "class-0",
    1: "class-1",
    2: "class-2",
    3: "class-3"
}

# Sidebar inputs
st.sidebar.markdown("### 📥 Video Input Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload a classroom learning clip (max 10s recommended)", 
    type=["mp4", "avi", "mov"]
)

st.sidebar.markdown("### ⚙️ Model Setup")
# Paths to checkpoints (relative to project root for robustness)
checkpoint_dir = Path(project_root) / "checkpoints"

def resolve_checkpoint(name):
    # Try best first, then final
    best_path = checkpoint_dir / f"{name}_best.pth"
    final_path = checkpoint_dir / f"{name}_final.pth"
    
    if best_path.exists():
        return str(best_path), f"{name}_best.pth"
    elif final_path.exists():
        return str(final_path), f"{name}_final.pth"
    return None, None

stgcn_ckpt, stgcn_loaded_name = resolve_checkpoint("stgcn")
bigru_ckpt, bigru_loaded_name = resolve_checkpoint("bigru")
convnext_ckpt, convnext_loaded_name = resolve_checkpoint("convnext")

# Load models and check if they exist
stgcn_ok = stgcn_ckpt is not None
bigru_ok = bigru_ckpt is not None
convnext_ok = convnext_ckpt is not None

st.sidebar.markdown("#### Model Status:")
if stgcn_ok:
    st.sidebar.success(f"ST-GCN Checkpoint: 🟢 Loaded (`{stgcn_loaded_name}`)")
else:
    st.sidebar.error("ST-GCN Checkpoint: 🔴 Not Found (using dummy weights)")
    
if bigru_ok:
    st.sidebar.success(f"Bi-GRU Checkpoint: 🟢 Loaded (`{bigru_loaded_name}`)")
else:
    st.sidebar.error("Bi-GRU Checkpoint: 🔴 Not Found (using dummy weights)")

if convnext_ok:
    st.sidebar.success(f"ConvNeXt1D Checkpoint: 🟢 Loaded (`{convnext_loaded_name}`)")
else:
    st.sidebar.error("ConvNeXt1D Checkpoint: 🔴 Not Found (using dummy weights)")

if not (stgcn_ok and bigru_ok and convnext_ok):
    st.sidebar.warning("Some models are missing trained weights. Running with randomly initialized models for preview/demonstration. Please train models first using training scripts.")

# Main app columns
col_video, col_results = st.columns([1, 1])

with col_video:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h4>📹 Video Source Preview</h4>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Save uploaded file locally to read via opencv
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Display video player
        st.video(tfile.name)
        st.success("Video successfully loaded. Press 'Analyze Engagement' in the sidebar to process.")
        
        # Trigger analysis
        analyze_button = st.sidebar.button("🧠 Analyze Engagement", use_container_width=True)
    else:
        st.info("Please upload a video file in the sidebar to begin analysis.")
        analyze_button = False
    st.markdown('</div>', unsafe_allow_html=True)

with col_results:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h4>🧠 Analysis & Predictions</h4>', unsafe_allow_html=True)
    
    if analyze_button and uploaded_file is not None:
        # Progress indicator
        progress_text = "Step 1/3: Extracting FaceMesh landmarks..."
        progress_bar = st.progress(0, text=progress_text)
        
        # 1. Feature Extraction
        try:
            landmarks = extract_features(tfile.name)
            progress_bar.progress(33, text="Step 2/3: Preprocessing landmark sequence...")
            
            # Pad/Truncate landmarks to 300 frames
            max_seq_len = 300
            T, V, C = landmarks.shape
            
            if T == 0:
                st.error("No faces detected in the video! Please ensure the student is facing the camera in a well-lit environment.")
                st.markdown('</div>', unsafe_allow_html=True)
                os.unlink(tfile.name)
                st.stop()
                
            if T < max_seq_len:
                pad_width = ((0, max_seq_len - T), (0, 0), (0, 0))
                landmarks = np.pad(landmarks, pad_width, mode='constant', constant_values=0.0)
            elif T > max_seq_len:
                landmarks = landmarks[:max_seq_len]
                
            progress_bar.progress(66, text="Step 3/3: Running ensemble inference models...")
            
            # Load Ensemble classifier
            ensemble = EnsembleClassifier(
                stgcn_ckpt=stgcn_ckpt if stgcn_ok else None,
                bigru_ckpt=bigru_ckpt if bigru_ok else None,
                convnext_ckpt=convnext_ckpt if convnext_ok else None,
                device="cpu"
            )
            
            # Run inference
            # landmarks tensor shape: (300, 468, 2) -> add batch dimension (1, 300, 468, 2)
            input_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
            
            # Get probabilities
            avg_probs = ensemble(input_tensor).squeeze(0).numpy() # shape (4,)
            pred_class = int(np.argmax(avg_probs))
            
            # Clear progress
            progress_bar.empty()
            
            # Display Prediction Card
            st.markdown("##### Predicted Engagement Level:")
            style_class = CLASS_STYLES[pred_class]
            st.markdown(f"""
            <div class="prediction-card {style_class} pulse">
                <h2 style='margin:0; font-size:1.8rem;'>{CLASS_MAP[pred_class]}</h2>
                <div style='font-size:0.95rem; margin-top:8px; opacity:0.85;'>
                    Confidence Score: {avg_probs[pred_class]*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Plot class probabilities
            st.write("")
            st.markdown("##### Prediction Confidence Distribution")
            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ['#f87171', '#fb923c', '#34d399', '#818cf8']
            bars = ax.barh(list(CLASS_MAP.values()), avg_probs, color=colors, height=0.55)
            ax.set_xlim(0, 1.0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='both', which='both', length=0)
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width*100:.1f}%', 
                        va='center', ha='left', fontsize=9, color='#cbd5e1')
                        
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            ax.xaxis.label.set_color('#cbd5e1')
            ax.tick_params(colors='#cbd5e1')
            st.pyplot(fig)
            
            # Sequence Info
            st.markdown("##### Landmark Sequence Details:")
            st.markdown(f"""
            - **Original frames**: {T}
            - **Model input shape**: `(1, {max_seq_len}, 468, 2)`
            - **Detected Landmark Coordinates**: Normalized `[X, Y]` coordinates
            """)
            
            # Delete local temp video
            os.unlink(tfile.name)
            
        except Exception as e:
            st.error(f"Failed to process video: {e}")
            if os.path.exists(tfile.name):
                os.unlink(tfile.name)
    else:
        st.write("Upload a video and press **Analyze Engagement** in the sidebar to display engagement classification.")
    st.markdown('</div>', unsafe_allow_html=True)

# Add a section about the methodology
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<h3>🔬 Project Architecture & Methodology</h3>', unsafe_allow_html=True)
st.markdown("""
This AI model utilizes a **heterogeneous deep learning ensemble** combining graph neural networks, sequential models, and CNN architectures for maximum robustness:
1. **MediaPipe FaceMesh**: Extracts 468 2D facial landmarks dynamically tracking head pose, gaze direction, eye blinks, and micro-expressions.
2. **ST-GCN (Spatial-Temporal Graph Convolutional Network)**: Models the spatial structure of face landmarks and the temporal sequence dynamics concurrently.
3. **Bi-GRU (Bidirectional GRU)**: Learns sequential patterns over the 300-frame video sequence from both forward and backward time steps.
4. **ConvNeXt1D**: Adapts modern CNN components (large kernel depthwise convs, LayerNorm, GELU) to process temporal features hierarchies.
5. **Soft Voting Ensemble**: Averages predictions (probabilities) from all three architectures to form the final high-confidence prediction.
""")
st.markdown('</div>', unsafe_allow_html=True)
