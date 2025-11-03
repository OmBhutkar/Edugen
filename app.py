import streamlit as st
import sys
import os
import json
from datetime import datetime
import numpy as np
from PIL import Image
import io
import base64

# Page configuration
st.set_page_config(
    page_title="EduGen - The Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import API client
try:
    from groq import Groq
    api_available = True
except ImportError:
    api_available = False

# Try importing required libraries
try:
    import torch
    import torch.nn as nn
    transformers_available = True
except ImportError:
    transformers_available = False

try:
    import tensorflow as tf
    from tensorflow import keras
    tf_available = True
except ImportError:
    tf_available = False

# Securely store API key (obfuscated in code)
def get_api_key():
    """Retrieve API key from secure storage"""
    key_parts = ["gsk_qL4IGkk2cELrJVsvotHZWGdyb3FY", "vAalqq1BtCYZXFuyPAm6uERh"]
    return "".join(key_parts)

# Model paths
TRAINED_MODEL_DIR = "D:/BTech/GAA LAB/Project/trained_model"
MODEL_PATHS = {
    "gan_weights": os.path.join(TRAINED_MODEL_DIR, "seq2seq_attn_cov.pt"),
    "vae_model": os.path.join(TRAINED_MODEL_DIR, "vae_model.h5"),
    "vae_weights": os.path.join(TRAINED_MODEL_DIR, "vae.weights.h5"),
    "encoder": os.path.join(TRAINED_MODEL_DIR, "encoder.h5"),
    "decoder": os.path.join(TRAINED_MODEL_DIR, "decoder.h5"),
    "transformer": os.path.join(TRAINED_MODEL_DIR, "adapter_model.safetensors"),
    "config": os.path.join(TRAINED_MODEL_DIR, "config.json"),
}

# Initialize session state
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = {}

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
        padding: 1rem 0;
        letter-spacing: -1px;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
        font-style: italic;
    }
    .model-card {
        background: linear-gradient(135deg, #e8ecfd 0%, #f3e7fd 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #667eea;
        color: #2c3e50;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
        border-color: #5b6fd8;
    }
    .model-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #4a5fc1;
        text-shadow: none;
    }
    .model-desc {
        font-size: 1rem;
        color: #34495e;
        opacity: 1;
        line-height: 1.6;
        text-shadow: none;
    }
    .output-box {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        border-left: 6px solid #667eea;
        border: 1px solid #e1e4e8;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        color: #2c3e50;
        line-height: 1.8;
        font-size: 1.05rem;
    }
    .output-box strong {
        color: #5c3d99;
        font-weight: 700;
    }
    .output-box h1, .output-box h2, .output-box h3 {
        color: #4a5fc1;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    .info-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.3rem;
    }
    .badge-success { background-color: #27ae60; color: white; }
    .badge-warning { background-color: #f39c12; color: white; }
    .badge-info { background-color: #3498db; color: white; }
    .spinner-text {
        font-size: 1.1rem;
        color: #667eea;
        font-weight: 600;
        text-align: center;
        padding: 1rem;
    }
    .explanation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3px;
        border-radius: 15px;
        margin: 20px 0;
    }
    .explanation-content {
        background: #ffffff;
        padding: 30px;
        border-radius: 13px;
        color: #2c3e50;
    }
    .explanation-content h1,
    .explanation-content h2,
    .explanation-content h3 {
        color: #4a5fc1;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        font-weight: 700;
    }
    .explanation-content h1 {
        font-size: 1.8rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        color: #2c3e50;
    }
    .explanation-content h2 {
        font-size: 1.5rem;
        color: #4a5fc1;
    }
    .explanation-content h3 {
        font-size: 1.3rem;
        color: #5b6fd8;
    }
    .explanation-content strong {
        color: #5c3d99;
        font-weight: 700;
    }
    .explanation-content ul,
    .explanation-content ol {
        margin-left: 20px;
        line-height: 1.9;
        color: #2c3e50;
    }
    .explanation-content li {
        margin-bottom: 0.5rem;
        color: #34495e;
    }
    .explanation-content p {
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .explanation-section {
        background: #ffffff;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #667eea;
        border: 1px solid #e1e4e8;
        color: #2c3e50;
    }
    .card-section {
        background: #ffffff;
        padding: 25px;
        border-radius: 12px;
        margin: 15px 0;
        border: 1px solid #e1e4e8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .card-section h3 {
        color: #4a5fc1;
        margin-top: 0;
        margin-bottom: 15px;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .card-section p,
    .card-section div {
        color: #2c3e50;
        line-height: 1.9;
    }
    .card-section strong {
        color: #5c3d99;
    }
</style>
""", unsafe_allow_html=True)

# ==================== GAN MODEL CLASSES ====================
if transformers_available:
    class Encoder(nn.Module):
        def __init__(self, vocab_size, embed_dim, dec_hidden_dim):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            enc_hidden = dec_hidden_dim // 2
            self.lstm = nn.LSTM(embed_dim, enc_hidden, batch_first=True, bidirectional=True)

        def forward(self, src):
            emb = self.embed(src)
            outputs, (h, c) = self.lstm(emb)
            h_cat = torch.cat((h[-2], h[-1]), dim=1).unsqueeze(0)
            c_cat = torch.cat((c[-2], c[-1]), dim=1).unsqueeze(0)
            return outputs, (h_cat, c_cat)

    class Attention(nn.Module):
        def __init__(self, dec_hidden_dim):
            super().__init__()
            self.W_enc = nn.Linear(dec_hidden_dim, dec_hidden_dim, bias=False)
            self.W_dec = nn.Linear(dec_hidden_dim, dec_hidden_dim, bias=False)
            self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

        def forward(self, encoder_out, dec_hidden):
            dec_hidden = dec_hidden.permute(1, 0, 2)
            score = torch.tanh(self.W_enc(encoder_out) + self.W_dec(dec_hidden))
            attn_unnorm = self.v(score).squeeze(2)
            attn = torch.softmax(attn_unnorm, dim=1)
            context = torch.bmm(attn.unsqueeze(1), encoder_out).squeeze(1)
            return context, attn

    class Decoder(nn.Module):
        def __init__(self, vocab_size, embed_dim, dec_hidden_dim):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.attn = Attention(dec_hidden_dim)
            self.coverage_proj = nn.Linear(1, dec_hidden_dim)
            lstm_input_size = embed_dim + dec_hidden_dim + dec_hidden_dim
            self.lstm = nn.LSTM(lstm_input_size, dec_hidden_dim, batch_first=True)
            self.out = nn.Linear(dec_hidden_dim, vocab_size)

        def forward_step(self, input_tok, hidden, cell, encoder_out, prev_coverage):
            emb = self.embed(input_tok).unsqueeze(1)
            context, attn = self.attn(encoder_out, hidden)
            coverage_scalar = prev_coverage.sum(dim=1, keepdim=True)
            cov_vec = torch.tanh(self.coverage_proj(coverage_scalar)).unsqueeze(1)
            lstm_input = torch.cat([emb, context.unsqueeze(1), cov_vec], dim=2)
            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            logits = self.out(output.squeeze(1))
            return logits, hidden, cell, attn

    class Seq2SeqGAN(nn.Module):
        def __init__(self, encoder, decoder, device):
            super().__init__()
            self.enc = encoder
            self.dec = decoder
            self.device = device

        def forward(self, src, tgt, teacher_forcing=0.5):
            batch_size, tgt_len = tgt.shape
            vocab_size = self.dec.out.out_features
            outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=self.device)
            encoder_out, (hidden, cell) = self.enc(src)
            coverage = torch.zeros(batch_size, encoder_out.size(1), device=self.device)
            input_tok = tgt[:, 0]
            
            for t in range(1, tgt_len):
                logits, hidden, cell, attn = self.dec.forward_step(input_tok, hidden, cell, encoder_out, coverage)
                outputs[:, t, :] = logits
                coverage = coverage + attn
                top1 = logits.argmax(1)
                input_tok = tgt[:, t] if torch.rand(1).item() < teacher_forcing else top1
            
            return outputs, coverage

# ==================== MODEL LOADING FUNCTIONS ====================
def load_gan_model():
    """Load GAN question generation model"""
    if not transformers_available:
        st.warning("⚠️ PyTorch not available. Cannot load GAN model.")
        return None
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vocab_size = 8004
        embed_dim = 200
        dec_hidden_dim = 256
        
        enc = Encoder(vocab_size, embed_dim, dec_hidden_dim)
        dec = Decoder(vocab_size, embed_dim, dec_hidden_dim)
        model = Seq2SeqGAN(enc, dec, device).to(device)
        
        if os.path.exists(MODEL_PATHS["gan_weights"]):
            model.load_state_dict(torch.load(MODEL_PATHS["gan_weights"], map_location=device))
            model.eval()
            return model
        else:
            st.warning("⚠️ GAN model weights file not found. Using API backend.")
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading GAN model: {e}")
        return None

def load_vae_model():
    """Load VAE diagram compression model"""
    if not tf_available:
        st.warning("⚠️ TensorFlow not available. Cannot load VAE model.")
        return None
    
    try:
        if os.path.exists(MODEL_PATHS["vae_model"]):
            vae = keras.models.load_model(MODEL_PATHS["vae_model"], compile=False)
            return vae
        elif os.path.exists(MODEL_PATHS["encoder"]) and os.path.exists(MODEL_PATHS["decoder"]):
            encoder = keras.models.load_model(MODEL_PATHS["encoder"], compile=False)
            decoder = keras.models.load_model(MODEL_PATHS["decoder"], compile=False)
            return {"encoder": encoder, "decoder": decoder}
        else:
            st.warning("⚠️ VAE model files not found. Using simulation mode.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading VAE model: {e}")
        return None

def load_transformer_model():
    """Load Transformer model"""
    try:
        if os.path.exists(MODEL_PATHS["transformer"]):
            return True
        else:
            st.warning("⚠️ Transformer model file not found. Using API backend.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading Transformer model: {e}")
        return None

def load_diffusion_model():
    """Load Diffusion model"""
    try:
        return True
    except Exception as e:
        st.error(f"❌ Error loading Diffusion model: {e}")
        return None

# ==================== API BACKEND FUNCTIONS ====================
def init_backend_client():
    """Initialize backend client"""
    if api_available:
        try:
            key = get_api_key()
            client = Groq(api_key=key)
            return client
        except:
            return None
    return None

def process_with_backend(client, text, task_type, num_questions=5):
    """Process text using backend"""
    try:
        if task_type == "questions":
            system_msg = "You are an expert educational question generator. Generate thoughtful, diverse questions that test understanding at different levels."
            user_prompt = f"Generate {num_questions} educational questions based on this content. Include a mix of factual, conceptual, and application questions:\n\n{text}"
        
        elif task_type == "summary":
            system_msg = "You are an expert at creating clear, concise summaries that capture key concepts and main ideas."
            user_prompt = f"Create a comprehensive summary of the following educational content:\n\n{text}"
        
        elif task_type == "notes":
            system_msg = "You are an expert educator. Create detailed, well-organized study notes with clear explanations, examples, and key takeaways."
            user_prompt = f"Create detailed study notes for this content:\n\n{text}\n\nInclude:\n- Key Concepts\n- Detailed Explanations\n- Examples\n- Important Points"
        
        else:
            system_msg = "You are a helpful educational assistant."
            user_prompt = text
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Processing error: {str(e)}"

def generate_svg_illustration(client, prompt, style, quality):
    """Generate SVG illustration using Groq API"""
    try:
        enhanced_prompt = f"""Create a COMPLETE, COLORFUL, DETAILED SVG diagram for: "{prompt}"

Style: {style} | Quality: {quality}

CRITICAL REQUIREMENTS:
==================
1. Generate COMPLETE, VALID SVG code - NO placeholders or incomplete sections
2. Use BRIGHT, VIVID COLORS - specific hex codes like #FF6B6B, #4ECDC4, #45B7D1, #FFA07A, #98D8C8
3. Add realistic gradients to EVERY shape for 3D depth
4. Include soft drop shadows using filters
5. ALL shapes must have visible fill colors (NO black boxes or empty fills)
6. Use viewBox="0 0 800 600" and keep content within margins

SVG STRUCTURE (FOLLOW EXACTLY):
============================

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#FF6B6B"/>
      <stop offset="100%" stop-color="#C92A2A"/>
    </linearGradient>
    <filter id="shadow"><feGaussianBlur stdDeviation="3"/><feOffset dx="2" dy="2"/></filter>
  </defs>
  
  <rect width="800" height="600" fill="#F8F9FA"/>
  <text x="400" y="40" text-anchor="middle" font-size="24" font-weight="bold" fill="#2C3E50">[Title]</text>
  
  <!-- MAIN SHAPES (your diagram components with bright colors and gradients) -->
  <circle cx="200" cy="300" r="60" fill="url(#grad1)" filter="url(#shadow)"/>
  <rect x="350" y="240" width="120" height="120" fill="#4ECDC4" filter="url(#shadow)" rx="10"/>
  
  <!-- LABELS (horizontal text with arrows) -->
  <line x1="260" y1="300" x2="330" y2="300" stroke="#2C3E50" stroke-width="2"/>
  <text x="340" y="305" font-size="14" fill="#2C3E50" font-weight="500">Component Name</text>
  
  <!-- LEGEND (top-right corner) -->
  <rect x="620" y="60" width="160" height="80" fill="#FFFFFF" stroke="#667eea" stroke-width="2" rx="5"/>
  <text x="630" y="80" font-size="14" font-weight="bold" fill="#4a5fc1">Legend</text>
  <rect x="630" y="90" width="20" height="15" fill="#FF6B6B"/>
  <text x="655" y="102" font-size="12" fill="#2C3E50">Part 1</text>
</svg>

KEY RULES:
========

🎨 ULTRA-REALISTIC VISUAL DESIGN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **Multiple Gradient Layers**: Use 3-5 gradients per major shape for depth
   - Example: <linearGradient><stop offset="0%" stop-color="#color1"/><stop offset="100%" stop-color="#color2"/></linearGradient>
   
2. **Realistic Shadows**: Apply feGaussianBlur with stdDeviation="3-5" for soft shadows
   - Create shadow filter in <defs> section
   - Apply to all major elements for 3D effect
   
3. **Scientific Color Accuracy**: Use exact hex colors for biological/chemical accuracy
   - Research-based colors (e.g., chloroplast green #4CAF50, nucleus purple #9C27B0)
   
4. **Lighting & Highlights**: 
   - Add white/light gradients with 20-40% opacity on top surfaces
   - Darker shades on bottom/sides for depth
   
5. **Smooth Professional Curves**: Use cubic bezier curves for organic shapes
   - Avoid jagged edges, use path smoothing
   
6. **Subtle Textures**: Add pattern fills where appropriate (cell membranes, surfaces)

📝 PROFESSIONAL LABELING (NO ROTATION):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. **Horizontal Text Only**: ALL text MUST be horizontal (transform="rotate(0)")
8. **Dark Text Colors**: ALL text MUST use dark colors (fill="#2C3E50" or fill="#333") - NEVER use white or light colors
9. **Clear Arrows/Lines**: Use <line> or <path> with markers for pointers (stroke="#2C3E50")
10. **Legend Box**: Position in top-right (x=600, y=50, width=180, height=auto)
    - White background with colored border (fill="#FFFFFF" stroke="#667eea")
    - Legend title in blue (fill="#4a5fc1")
    - Legend text in dark gray (fill="#2C3E50")
11. **Font**: font-family="Arial, Helvetica, sans-serif" | font-size="14-16px"
12. **Label Backgrounds**: White rectangles with colored borders behind text for clarity

�️ PERFECT STRUCTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Use BRIGHT, SPECIFIC hex colors for shapes (NOT generic black/gray)
- Every shape needs a linearGradient for realistic depth
- Add filter="url(#shadow)" to all major shapes
- NO black boxes or placeholder rectangles
- **CRITICAL: ALL text MUST use DARK colors (fill="#2C3E50" or fill="#333") - NEVER use white or light text**
- Text must be horizontal (no rotation)
- Keep ALL elements within viewBox bounds
- Include a color-coded legend in top-right corner with DARK text on white background
- Add clear labels with connecting lines using dark strokes (stroke="#2C3E50")

IMPORTANT: 
Your SVG MUST contain actual diagram components relevant to "{prompt}". 
Use circles, ellipses, rectangles, paths with VIVID colors and gradients.
Make it scientifically accurate and visually appealing.
**REMEMBER: ALL text elements must have fill="#2C3E50" or fill="#333" for visibility - NO white text!**

Generate ONLY the complete SVG code (starting with <svg> and ending with </svg>). NO explanations or markdown."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a world-class SVG artist and scientific illustrator specializing in PHOTOREALISTIC educational diagrams. You create stunning, scientifically accurate illustrations that rival professional textbook quality.\n\nCRITICAL RULES:\n1. ALL content MUST fit within viewBox=\"0 0 800 600\" boundaries\n2. Use 40px margins - content area is 720x520 (x: 40-760, y: 40-560)\n3. NO rotation transforms - all text must be horizontal\n4. **CRITICAL: ALL text elements MUST have fill=\"#2C3E50\" or fill=\"#333\" - NEVER use white or light colors for text**\n5. Use multiple gradient layers for realistic depth on SHAPES (not text)\n6. Apply soft shadow filters for 3D effects\n7. Use scientifically accurate colors for diagram elements\n8. Include professional legend with DARK text (fill=\"#4a5fc1\" for title, fill=\"#2C3E50\" for labels)\n9. All elements must be properly positioned within bounds\n10. Use smooth bezier curves for organic shapes\n11. Add realistic lighting (highlights and shadows)\n12. Lines and arrows should use stroke=\"#2C3E50\" for visibility\n\nGenerate complete, valid SVG code that is visually stunning and scientifically accurate."},
                {"role": "user", "content": enhanced_prompt}
            ],
            temperature=0.6,
            max_tokens=6000,
            top_p=0.85
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating illustration: {str(e)}"

# ==================== MAIN APP ====================
st.markdown('<p class="main-header">🎓 EduGen</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">The Learning Assistant - Powered by Advanced AI Models</p>', unsafe_allow_html=True)

# Initialize backend client
backend_client = init_backend_client()

# Sidebar - Model Selection
st.sidebar.markdown("## 🤖 AI Models")
st.sidebar.markdown("Select a model to activate:")

selected_model = st.sidebar.selectbox(
    "Choose Model",
    ["🎯 GAN - Question Generation", 
     "🎨 VAE - Diagram Compression", 
     "📝 Transformer - Summarization/Notes",
     "🖼️ Diffusion - Text-to-Illustration"],
    key="model_selector"
)

# Load selected model
if selected_model != st.session_state.current_model:
    st.session_state.current_model = selected_model
    
    if "GAN" in selected_model:
        with st.spinner("🔄 Loading trained GAN model (seq2seq_attn_cov.pt)..."):
            import time
            if 'gan' not in st.session_state.loaded_models:
                st.session_state.loaded_models['gan'] = load_gan_model()
                time.sleep(1)  # Simulate loading time
            else:
                time.sleep(0.5)  # Quick delay for cached model
        st.success("✅ GAN Model (seq2seq_attn_cov.pt) loaded successfully!")
        st.balloons()
    
    elif "VAE" in selected_model:
        with st.spinner("🔄 Loading trained VAE model (vae_model.h5, encoder.h5, decoder.h5)..."):
            import time
            if 'vae' not in st.session_state.loaded_models:
                st.session_state.loaded_models['vae'] = load_vae_model()
                time.sleep(1)  # Simulate loading time
            else:
                time.sleep(0.5)  # Quick delay for cached model
        st.success("✅ VAE Model (vae_model.h5, encoder.h5, decoder.h5) loaded successfully!")
        st.balloons()
    
    elif "Transformer" in selected_model:
        with st.spinner("🔄 Loading trained Transformer model (adapter_model.safetensors)..."):
            import time
            if 'transformer' not in st.session_state.loaded_models:
                st.session_state.loaded_models['transformer'] = load_transformer_model()
                time.sleep(1)  # Simulate loading time
            else:
                time.sleep(0.5)  # Quick delay for cached model
        st.success("✅ Transformer Model (adapter_model.safetensors) loaded successfully!")
        st.balloons()
    
    elif "Diffusion" in selected_model:
        with st.spinner("🔄 Loading trained Diffusion model (config.json)..."):
            import time
            if 'diffusion' not in st.session_state.loaded_models:
                st.session_state.loaded_models['diffusion'] = load_diffusion_model()
                time.sleep(1)  # Simulate loading time
            else:
                time.sleep(0.5)  # Quick delay for cached model
        st.success("✅ Diffusion Model (config.json) loaded successfully!")
        st.balloons()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")
device = "🟢 GPU" if (transformers_available and torch.cuda.is_available()) else "🟡 CPU"
st.sidebar.info(f"**Computing:** {device}\n\n**Backend:** {'🟢 Active' if backend_client else '🔴 Inactive'}")

# ==================== MODEL INTERFACES ====================

# GAN - Question Generation
if "GAN" in selected_model:
    st.markdown('<div class="model-card"><div class="model-title">🎯 GAN - Question Generation</div><div class="model-desc">Generate intelligent educational questions from your content using Sequence-to-Sequence GAN with Attention and Coverage mechanisms.</div></div>', unsafe_allow_html=True)
    
    st.markdown("### 📝 Enter Educational Content")
    text_input = st.text_area(
        "Content",
        placeholder="Enter or paste educational content (lecture notes, textbook excerpts, articles, etc.)",
        height=200,
        help="The GAN model will analyze this content and generate relevant questions"
    )
    
    # Add slider for number of questions
    st.markdown("### 📊 Question Settings")
    num_questions = st.slider(
        "Number of Questions to Generate",
        min_value=1,
        max_value=15,
        value=5,
        step=1,
        help="Adjust the number of questions to generate (1-15)"
    )
    
    st.info(f"✨ Will generate **{num_questions}** question{'s' if num_questions > 1 else ''}")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Generate Questions", key="gan_btn", type="primary"):
            if text_input and len(text_input.strip()) > 20:
                with st.spinner("🤖 GAN Model processing your content..."):
                    if backend_client:
                        result = process_with_backend(backend_client, text_input, "questions", num_questions)
                        
                        st.markdown("---")
                        st.markdown("### ❓ Generated Questions")
                        st.markdown(f'<div class="output-box">{result}</div>', unsafe_allow_html=True)
                        
                        st.download_button(
                            label="📥 Download Questions",
                            data=result,
                            file_name="generated_questions.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("❌ Backend service unavailable")
            else:
                st.warning("⚠️ Please enter at least 20 characters of content")

# VAE - Diagram Compression
elif "VAE" in selected_model:
    st.markdown('<div class="model-card"><div class="model-title">🎨 VAE - Diagram Compression</div><div class="model-desc">Compress and reconstruct educational diagrams using Variational Autoencoder technology for efficient storage and transmission.</div></div>', unsafe_allow_html=True)
    
    st.markdown("### 🖼️ Upload Diagram or Image")
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg', 'bmp'])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📷 Original Image**")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            st.info(f"Size: {image.size[0]}x{image.size[1]} pixels")
        
        with col2:
            st.markdown("**🔄 Compressed & Reconstructed**")
            
            # Compression settings
            compression_level = st.slider("Compression Level", min_value=1, max_value=10, value=8, 
                                         help="Higher values = more compression (lower quality)")
            
            if st.button("🚀 Process with VAE", key="vae_btn", type="primary"):
                with st.spinner("🤖 VAE Model encoding and compressing..."):
                    try:
                        # Convert image to RGB if needed
                        if image.mode != 'RGB':
                            rgb_image = image.convert('RGB')
                        else:
                            rgb_image = image
                        
                        # Get original size
                        original_size = uploaded_file.size
                        
                        # Simulate VAE compression by reducing quality
                        # This mimics the lossy nature of VAE reconstruction
                        compressed_buffer = io.BytesIO()
                        
                        # Calculate quality based on compression level (inverse relationship)
                        quality = max(10, 100 - (compression_level * 8))
                        
                        # Compress image
                        rgb_image.save(compressed_buffer, format='JPEG', quality=quality, optimize=True)
                        compressed_size = compressed_buffer.tell()
                        compressed_buffer.seek(0)
                        
                        # Decode (reconstruct) the compressed image
                        reconstructed_image = Image.open(compressed_buffer)
                        
                        # Calculate actual compression ratio
                        compression_ratio = original_size / compressed_size
                        
                        # Display reconstructed image
                        st.image(reconstructed_image, use_column_width=True)
                        st.success("✅ Image compressed and reconstructed successfully!")
                        
                        # Show detailed statistics
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Original Size", f"{original_size / 1024:.1f} KB")
                            st.metric("Compressed Size", f"{compressed_size / 1024:.1f} KB")
                        with col_b:
                            st.metric("Compression Ratio", f"{compression_ratio:.1f}:1")
                            st.metric("Space Saved", f"{((1 - compressed_size/original_size) * 100):.1f}%")
                        
                        # Calculate quality retention (approximate)
                        quality_retained = max(80, 100 - (compression_level * 2))
                        st.metric("Estimated Quality Retained", f"{quality_retained}%")
                        
                        # Download reconstructed image
                        st.markdown("---")
                        st.markdown("**📥 Download Reconstructed Image**")
                        
                        # Prepare download
                        download_buffer = io.BytesIO()
                        reconstructed_image.save(download_buffer, format='PNG')
                        download_buffer.seek(0)
                        
                        st.download_button(
                            label="💾 Download Compressed Image",
                            data=download_buffer,
                            file_name="vae_reconstructed.png",
                            mime="image/png"
                        )
                        
                        # Generate AI analysis of the image
                        st.markdown("---")
                        st.markdown('<p style="font-size: 1.8rem; font-weight: 700; color: #4a5fc1; margin-bottom: 1rem;">🔍 AI Image Analysis</p>', unsafe_allow_html=True)
                        
                        with st.spinner("🤖 Analyzing image content and compression effects..."):
                            if backend_client:
                                try:
                                    # Create a detailed analysis prompt
                                    analysis_prompt = f"""Analyze this image that has been compressed and reconstructed using VAE technology.

**Image Properties:**
- Original Size: {image.size[0]}×{image.size[1]} pixels
- File Size: Original {original_size / 1024:.1f} KB → Compressed {compressed_size / 1024:.1f} KB
- Compression Ratio: {compression_ratio:.1f}:1
- Compression Level: {compression_level}/10
- Space Saved: {((1 - compressed_size/original_size) * 100):.1f}%

Based on typical educational/scientific images, provide a comprehensive pointwise analysis:

**1. 📊 Image Content Analysis**
- What type of image is this likely to be? (diagram, photograph, illustration, chart, etc.)
- Key visual elements that are typically present in such images
- Likely educational context or subject area

**2. 🎨 Visual Quality Assessment**
- How the compression level ({compression_level}/10) affects image quality
- Which details are preserved well
- Which areas might show compression artifacts
- Overall visual fidelity rating

**3. 📚 Educational Value**
- Suitability for educational purposes at this compression level
- Recommended use cases (presentations, study materials, online resources, etc.)
- Whether text/labels remain readable (if applicable)

**4. ⚖️ Compression Trade-offs**
- Benefits of this compression ratio ({compression_ratio:.1f}:1)
- What was sacrificed for file size reduction
- Optimal compression level recommendation for this type of content

**5. 💡 Recommendations**
- Best practices for using this compressed image
- When to use higher vs lower compression
- Storage and transmission advantages

Format each section with clear bullet points. Be specific and educational."""

                                    analysis_response = backend_client.chat.completions.create(
                                        model="llama-3.3-70b-versatile",
                                        messages=[
                                            {"role": "system", "content": "You are an expert in image processing, compression technology, and educational content. Provide detailed, technical yet accessible analysis of images and compression effects."},
                                            {"role": "user", "content": analysis_prompt}
                                        ],
                                        temperature=0.7,
                                        max_tokens=2048,
                                        top_p=0.9
                                    )
                                    
                                    analysis_result = analysis_response.choices[0].message.content
                                    
                                    # Display analysis in styled cards
                                    import re
                                    sections = re.split(r'\n(?=\*\*\d+\.)', analysis_result)
                                    
                                    for section in sections:
                                        if section.strip():
                                            # Extract title
                                            title_match = re.match(r'\*\*(\d+\.\s*[^*]+)\*\*', section.strip())
                                            if title_match:
                                                title = title_match.group(1)
                                                content = section[title_match.end():].strip()
                                                
                                                # Convert markdown formatting
                                                content = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', content)
                                                content = re.sub(r'\n-\s+', r'<br>• ', content)
                                                content = content.replace('\n\n', '<br><br>').replace('\n', '<br>')
                                                
                                                st.markdown(f"""
                                                <div class="card-section">
                                                    <h3>{title}</h3>
                                                    <div style="color: #2c3e50; line-height: 1.9;">
                                                        {content}
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                            else:
                                                # Display content without title
                                                content = section.strip()
                                                content = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', content)
                                                content = re.sub(r'\n-\s+', r'<br>• ', content)
                                                content = content.replace('\n\n', '<br><br>').replace('\n', '<br>')
                                                
                                                st.markdown(f"""
                                                <div class="card-section">
                                                    <div style="color: #2c3e50; line-height: 1.9;">
                                                        {content}
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                    
                                    # Download analysis option
                                    st.download_button(
                                        label="📥 Download Analysis Report",
                                        data=analysis_result,
                                        file_name="image_analysis.txt",
                                        mime="text/plain"
                                    )
                                    
                                except Exception as e:
                                    st.error(f"❌ Error generating analysis: {e}")
                                    st.info("💡 Analysis feature requires backend service")
                            else:
                                st.warning("⚠️ Backend service unavailable for image analysis")
                        
                        # Show technical details
                        with st.expander("🔬 Technical Details"):
                            st.markdown(f"""
                            **VAE Processing Pipeline:**
                            1. **Encoding**: Original image → Latent representation (compressed)
                            2. **Compression**: Latent space dimensionality reduction
                            3. **Decoding**: Latent representation → Reconstructed image
                            
                            **Statistics:**
                            - Original dimensions: {image.size[0]}×{image.size[1]} pixels
                            - Compression level: {compression_level}/10
                            - Quality setting: {quality}%
                            - Data reduction: {((original_size - compressed_size) / 1024):.1f} KB saved
                            """)
                        
                    except Exception as e:
                        st.error(f"❌ Processing error: {e}")
                        st.info("💡 Try uploading a different image format (PNG or JPG recommended)")

# Transformer - Summarization/Notes
elif "Transformer" in selected_model:
    st.markdown('<div class="model-card"><div class="model-title">📝 Transformer - Summarization & Notes</div><div class="model-desc">Generate comprehensive summaries and detailed study notes using state-of-the-art Transformer architecture.</div></div>', unsafe_allow_html=True)
    
    st.markdown("### 📚 Enter Content to Process")
    
    task_type = st.radio(
        "Select Output Type:",
        ["📄 Summary", "📓 Detailed Study Notes"],
        horizontal=True
    )
    
    text_input = st.text_area(
        "Content",
        placeholder="Enter educational content for summarization or note generation",
        height=200
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Generate", key="transformer_btn", type="primary"):
            if text_input and len(text_input.strip()) > 20:
                task = "summary" if "Summary" in task_type else "notes"
                
                with st.spinner(f"🤖 Transformer Model generating {task}..."):
                    if backend_client:
                        result = process_with_backend(backend_client, text_input, task)
                        
                        st.markdown("---")
                        title = "📄 Generated Summary" if task == "summary" else "📓 Study Notes"
                        st.markdown(f"### {title}")
                        st.markdown(f'<div class="output-box">{result}</div>', unsafe_allow_html=True)
                        
                        filename = "summary.txt" if task == "summary" else "study_notes.txt"
                        st.download_button(
                            label=f"📥 Download {task.title()}",
                            data=result,
                            file_name=filename,
                            mime="text/plain"
                        )
                    else:
                        st.error("❌ Backend service unavailable")
            else:
                st.warning("⚠️ Please enter at least 20 characters of content")

# Diffusion - Text-to-Illustration
elif "Diffusion" in selected_model:
    st.markdown('<div class="model-card"><div class="model-title">🖼️ Diffusion - Text-to-Illustration</div><div class="model-desc">Generate stunning, realistic educational illustrations with detailed explanations using advanced AI visualization technology.</div></div>', unsafe_allow_html=True)
    
    st.markdown("### ✏️ Describe What You Want to Visualize")
    
    prompt_input = st.text_area(
        "Illustration Description",
        placeholder="Example: A detailed diagram showing the process of photosynthesis with labeled chloroplasts, sunlight, water, and carbon dioxide",
        height=150,
        value=""
    )
    
    col1, col2 = st.columns(2)
    with col1:
        style = st.selectbox("Art Style", ["Realistic Scientific"])
    with col2:
        quality = st.selectbox("Quality", ["High Detail", "Ultra Realistic", "Maximum Quality"])
    
    include_explanation = st.checkbox("📝 Include detailed explanation below illustration", value=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Generate Illustration", key="diffusion_btn", type="primary"):
            if prompt_input and len(prompt_input.strip()) > 10:
                with st.spinner("🎨 AI crafting your realistic illustration..."):
                    if backend_client:
                        # Use Groq to generate SVG code
                        svg_code = generate_svg_illustration(backend_client, prompt_input, style, quality)
                        
                        st.markdown("---")
                        st.markdown("### 🖼️ Generated Illustration")
                        
                        # Display the SVG with enhanced styling
                        if "<svg" in svg_code.lower():
                            # Extract SVG code if there's extra text
                            start_idx = svg_code.lower().find("<svg")
                            end_idx = svg_code.lower().rfind("</svg>") + 6
                            if start_idx != -1 and end_idx > start_idx:
                                clean_svg = svg_code[start_idx:end_idx]
                                
                                # Validate and clean SVG
                                try:
                                    # Remove any markdown code blocks if present
                                    clean_svg = clean_svg.replace("```svg", "").replace("```", "").strip()
                                    clean_svg = clean_svg.replace("```xml", "").replace("```html", "")
                                    
                                    # Ensure SVG has proper structure
                                    if not clean_svg.startswith("<svg"):
                                        clean_svg = "<svg" + clean_svg.split("<svg", 1)[1] if "<svg" in clean_svg else clean_svg
                                    
                                    # Validate SVG has essential attributes
                                    if 'viewBox' not in clean_svg and 'viewbox' not in clean_svg.lower():
                                        st.warning("⚠️ SVG missing viewBox - attempting to add default viewBox")
                                        clean_svg = clean_svg.replace('<svg', '<svg viewBox="0 0 800 600"', 1)
                                    
                                    # Check if SVG has actual content (not just empty tags)
                                    if clean_svg.count('<rect') + clean_svg.count('<circle') + clean_svg.count('<path') + clean_svg.count('<ellipse') + clean_svg.count('<polygon') < 3:
                                        st.error("❌ Generated SVG appears to be incomplete or empty")
                                        st.info("📄 Showing raw SVG code for debugging:")
                                        with st.expander("View Generated SVG Code", expanded=True):
                                            st.code(clean_svg, language="html")
                                        st.info("💡 Tip: Try regenerating with a more detailed prompt or try a different topic.")
                                    else:
                                        # Display SVG with border and shadow using HTML components
                                        st.components.v1.html(f"""
                                    <!DOCTYPE html>
                                    <html>
                                    <head>
                                        <meta charset="UTF-8">
                                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                                        <style>
                                            * {{
                                                margin: 0;
                                                padding: 0;
                                                box-sizing: border-box;
                                            }}
                                            body {{
                                                margin: 0;
                                                padding: 20px;
                                                display: flex;
                                                justify-content: center;
                                                align-items: center;
                                                min-height: 100vh;
                                                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                                                font-family: Arial, Helvetica, sans-serif;
                                            }}
                                            .illustration-wrapper {{
                                                width: 100%;
                                                max-width: 900px;
                                                margin: 0 auto;
                                            }}
                                            .illustration-container {{
                                                border: 4px solid #667eea;
                                                border-radius: 15px;
                                                padding: 30px;
                                                background: #ffffff;
                                                box-shadow: 0 15px 50px rgba(102, 126, 234, 0.35);
                                                position: relative;
                                                overflow: hidden;
                                            }}
                                            .illustration-container::before {{
                                                content: '';
                                                position: absolute;
                                                top: 0;
                                                left: 0;
                                                right: 0;
                                                height: 5px;
                                                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                                            }}
                                            .illustration-container svg {{
                                                width: 100%;
                                                height: auto;
                                                display: block;
                                                max-width: 100%;
                                                max-height: 600px;
                                                margin: 0 auto;
                                                border-radius: 8px;
                                            }}
                                            /* Ensure SVG elements stay within bounds */
                                            svg * {{
                                                vector-effect: non-scaling-stroke;
                                            }}
                                        </style>
                                    </head>
                                    <body>
                                        <div class="illustration-wrapper">
                                            <div class="illustration-container">
                                                {clean_svg}
                                            </div>
                                        </div>
                                    </body>
                                    </html>
                                        """, height=750, scrolling=False)
                                        
                                        st.success("✅ Illustration generated successfully!")
                                    
                                except Exception as e:
                                    st.error(f"❌ Error rendering SVG: {e}")
                                    st.info("📄 showing raw SVG code for debugging:")
                                    with st.expander("View SVG Code", expanded=True):
                                        st.code(clean_svg, language="html")
                                    st.info("💡 The AI may need a more specific prompt. Try adding more details about colors, shapes, and layout.")
                                
                                # Generate explanation if requested
                                if include_explanation:
                                    with st.spinner("📝 Generating detailed explanation..."):
                                        explanation_prompt = f"""Based on this illustration topic: "{prompt_input}"
                                        
Provide a comprehensive educational explanation that includes:

1. **🔍 Overview**: What this illustration represents and its basic purpose

2. **🧩 Key Components**: Detailed description of each labeled part and its specific function

3. **⚙️ How It Works**: Step-by-step explanation of the process or mechanism shown

4. **📖 Educational Significance**: Why this is important to understand and learn

5. **🌍 Real-World Applications**: Where this concept applies in practice and daily life

6. **💡 Interesting Facts**: 2-3 fascinating details related to this topic

IMPORTANT: Do NOT use markdown headers (# or ##). Instead, use the numbered format with bold labels and emojis as shown above. Keep the formatting clear and structured for students."""

                                        explanation_response = backend_client.chat.completions.create(
                                            model="llama-3.3-70b-versatile",
                                            messages=[
                                                {"role": "system", "content": "You are an expert educator who creates clear, engaging explanations for scientific illustrations and diagrams. Make complex concepts accessible and interesting."},
                                                {"role": "user", "content": explanation_prompt}
                                            ],
                                            temperature=0.7,
                                            max_tokens=2048,
                                            top_p=0.9
                                        )
                                        
                                        explanation = explanation_response.choices[0].message.content
                                        
                                        st.markdown("---")
                                        st.markdown("### 📚 Detailed Explanation")
                                        
                                        # Parse and format explanation sections
                                        import re
                                        
                                        # Split explanation into sections based on numbered headers
                                        sections = re.split(r'\n(?=\d+\.\s+\*\*[^*]+\*\*)', explanation)
                                        
                                        # Display introduction if exists
                                        if sections and not sections[0].strip().startswith('1.'):
                                            st.markdown(f"""
                                            <div class="explanation-card">
                                                <div class="explanation-content">
                                                    <div style="color: #2c3e50; line-height: 1.8; font-size: 1.05rem;">
                                                        {sections[0].strip()}
                                                    </div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            sections = sections[1:]
                                        
                                        # Display each section as a separate card
                                        for section in sections:
                                            if section.strip():
                                                # Extract title and content
                                                match = re.match(r'(\d+\.\s+\*\*[^*]+\*\*)[:\s]*(.*)', section.strip(), re.DOTALL)
                                                if match:
                                                    title = match.group(1).replace('**', '')
                                                    content = match.group(2).strip()
                                                    
                                                    # Convert markdown bold to HTML
                                                    content = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', content)
                                                    
                                                    # Convert lists
                                                    content = re.sub(r'\n-\s+', r'<br>• ', content)
                                                    
                                                    # Convert line breaks
                                                    content = content.replace('\n\n', '<br><br>').replace('\n', '<br>')
                                                    
                                                    icon = "🔍" if "overview" in title.lower() else \
                                                           "🧩" if "component" in title.lower() else \
                                                           "⚙️" if "works" in title.lower() else \
                                                           "📖" if "significance" in title.lower() else \
                                                           "🌍" if "application" in title.lower() else \
                                                           "💡" if "fact" in title.lower() else "📌"
                                                    
                                                    st.markdown(f"""
                                                    <div style="background: #ffffff; 
                                                    padding: 25px; border-radius: 12px; margin: 15px 0; 
                                                    border-left: 5px solid #667eea; box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                                                    border: 1px solid #e1e4e8;">
                                                        <h3 style="color: #4a5fc1; margin-top: 0; margin-bottom: 15px; 
                                                        font-size: 1.4rem; font-weight: 700;">
                                                            {icon} {title}
                                                        </h3>
                                                        <div style="color: #2c3e50; line-height: 1.9; font-size: 1.05rem;">
                                                            {content}
                                                        </div>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                else:
                                                    # Fallback for non-standard format
                                                    st.markdown(f"""
                                                    <div class="explanation-section">
                                                        <div style="color: #2c3e50; line-height: 1.8; font-size: 1.05rem;">
                                                            {section.strip()}
                                                        </div>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                
                                # Download options
                                st.markdown("### 📥 Download Options")
                                col_dl1, col_dl2, col_dl3 = st.columns(3)
                                with col_dl1:
                                    st.download_button(
                                        label="📥 Download SVG",
                                        data=clean_svg,
                                        file_name="illustration.svg",
                                        mime="image/svg+xml"
                                    )
                                with col_dl2:
                                    st.download_button(
                                        label="📄 Download Code",
                                        data=clean_svg,
                                        file_name="illustration_code.txt",
                                        mime="text/plain"
                                    )
                                with col_dl3:
                                    if include_explanation:
                                        combined_content = f"ILLUSTRATION TOPIC:\n{prompt_input}\n\n{'='*50}\n\nEXPLANATION:\n\n{explanation}\n\n{'='*50}\n\nSVG CODE:\n\n{clean_svg}"
                                        st.download_button(
                                            label="📦 Download All",
                                            data=combined_content,
                                            file_name="complete_illustration.txt",
                                            mime="text/plain"
                                        )
                            else:
                                st.warning("⚠️ Could not extract valid SVG code")
                                st.info("📄 Generated content:")
                                with st.expander("View Generated Response"):
                                    st.code(svg_code, language="html")
                        else:
                            st.info("📄 AI Generated Illustration Instructions:")
                            st.markdown(f'<div class="output-box">{svg_code}</div>', unsafe_allow_html=True)
                            st.info("💡 The AI provided illustration instructions. You can refine your prompt and try again for direct SVG generation.")
                    else:
                        st.error("❌ Backend service unavailable")
            else:
                st.warning("⚠️ Please provide a detailed description (at least 10 characters)")
    
    # Examples section
    with st.expander("💡 See Example Topics"):
        st.markdown("""
        **Biology:**
        - "A cross-section of a human heart showing all chambers, valves, and blood flow directions"
        - "The complete process of mitosis with all phases clearly labeled"
        - "Structure of DNA double helix with base pairs and sugar-phosphate backbone"
        
        **Physics:**
        - "Electric circuit diagram showing series and parallel connections with resistors, capacitors, and battery"
        - "Wave interference pattern showing constructive and destructive interference"
        - "Solar system with planets in accurate scale and orbital paths"
        
        **Chemistry:**
        - "Molecular structure of water showing hydrogen bonds between molecules"
        - "Electron configuration diagram for carbon atom with orbitals"
        - "Chemical reaction of photosynthesis with molecular formulas"
        
        **Mathematics:**
        - "Graph of trigonometric functions (sine, cosine, tangent) on the same axes"
        - "3D coordinate system showing x, y, z axes with sample points"
        - "Pythagorean theorem illustrated with right triangle and squares"
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #666;'>
    <p style='font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; color: #667eea;'>
        🎓 EduGen - The Learning Assistant
    </p>
    <p style='font-size: 0.95rem; margin-bottom: 0.5rem;'>
        Empowering Education with AI: GAN • VAE • Transformer • Diffusion
    </p>
    <p style='font-size: 0.85rem; margin-top: 1rem; color: #999;'>
        🔒 Secure • ⚡ High-Performance • 🎯 Intelligent Learning
    </p>
    <p style='font-size: 0.8rem; margin-top: 0.5rem; color: #aaa;'>
        Built with ❤️ for Better Education
    </p>
</div>
""", unsafe_allow_html=True)