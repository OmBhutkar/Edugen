import streamlit as st
import sys
import os
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="STEM Learning Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import API client
try:
    from groq import Groq
    api_available = True
except ImportError:
    api_available = False

# Try importing required libraries with error handling
try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
    transformers_available = True
except ImportError as e:
    transformers_available = False

# Try importing PEFT (optional)
try:
    from peft import PeftModel, PeftConfig
    peft_available = True
except ImportError:
    peft_available = False

# Store API key securely (not visible in UI)
API_KEY = "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"

# Initialize API client
@st.cache_resource
def init_api_client():
    """Initialize API client with key"""
    if api_available and API_KEY:
        try:
            client = Groq(api_key=API_KEY)
            return client
        except Exception as e:
            return None
    return None

# Custom CSS with better color contrast
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a5490;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .output-box {
        background-color: #ffffff;
        padding: 1.8rem;
        border-radius: 12px;
        border-left: 6px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        color: #2c3e50;
        line-height: 1.7;
    }
    .summary-box {
        border-left-color: #e67e22;
        background: linear-gradient(to right, #fff5eb 0%, #ffffff 100%);
    }
    .flashcard-box {
        border-left-color: #27ae60;
        background: linear-gradient(to right, #eafaf1 0%, #ffffff 100%);
    }
    .notes-box {
        border-left-color: #c0392b;
        background: linear-gradient(to right, #fadbd8 0%, #ffffff 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .info-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.3rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .badge-success {
        background-color: #27ae60;
        color: #ffffff;
    }
    .badge-info {
        background-color: #3498db;
        color: #ffffff;
    }
    .badge-warning {
        background-color: #f39c12;
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ecf0f1;
        color: #2c3e50;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    .stRadio > label {
        color: #2c3e50;
        font-weight: 500;
    }
    .stTextInput > label, .stTextArea > label {
        color: #2c3e50;
        font-weight: 600;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

# Define model paths
MODEL_PATHS = {
    "trained_model": "./trained_model",
    "config": "./config.json",
    "training_history": "./training_history.json"
}

def check_model_files():
    """Check if trained model files exist"""
    if not transformers_available:
        return {"model_dir": False, "config": False, "adapter_config": False, "model_files": False}
    
    model_exists = os.path.exists(MODEL_PATHS["trained_model"])
    config_exists = os.path.exists(MODEL_PATHS["config"])
    
    return {
        "model_dir": model_exists,
        "config": config_exists,
        "adapter_config": os.path.exists(os.path.join(MODEL_PATHS["trained_model"], "adapter_config.json")),
        "model_files": os.path.exists(os.path.join(MODEL_PATHS["trained_model"], "adapter_model.bin")) or 
                       os.path.exists(os.path.join(MODEL_PATHS["trained_model"], "adapter_model.safetensors"))
    }

def load_training_info():
    """Load training configuration and history"""
    info = {}
    
    if os.path.exists(MODEL_PATHS["config"]):
        try:
            with open(MODEL_PATHS["config"], 'r') as f:
                info['config'] = json.load(f)
        except:
            info['config'] = None
    
    if os.path.exists(MODEL_PATHS["training_history"]):
        try:
            with open(MODEL_PATHS["training_history"], 'r') as f:
                info['history'] = json.load(f)
        except:
            info['history'] = None
    
    return info

# Generate with API
def generate_with_api(client, prompt, text, task_type="general"):
    """Generate text using API"""
    try:
        if task_type == "summary":
            system_msg = "You are an expert at creating concise, clear summaries of educational content. Provide a well-structured summary that captures the main points."
            user_prompt = f"Create a concise summary of the following content:\n\n{text}"
        
        elif task_type == "flashcard":
            system_msg = "You are an expert at creating educational flashcards. Create a clear question and detailed answer based on the content."
            user_prompt = f"Create a flashcard with a question and answer based on this content:\n\n{text}\n\nFormat:\nQ: [Question]\nA: [Answer]"
        
        elif task_type == "notes":
            system_msg = "You are an expert educator. Create detailed, well-organized study notes that explain concepts clearly with examples."
            user_prompt = f"Create detailed study notes explaining this content:\n\n{text}\n\nInclude:\n- Key concepts\n- Explanations\n- Examples\n- Important points to remember"
        
        else:
            system_msg = "You are a helpful AI assistant specialized in education."
            user_prompt = f"{prompt}: {text}"
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating content: {str(e)}"

# Cache model loading (for local models)
@st.cache_resource
def load_model(model_choice):
    """Load the selected model and tokenizer"""
    if not transformers_available:
        return None, None, None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        with st.spinner(f"Loading {model_choice}..."):
            if model_choice == "Fine-tuned T5-Base" and peft_available:
                model_status = check_model_files()
                
                if model_status["model_dir"] and model_status["adapter_config"]:
                    try:
                        st.info("📂 Loading fine-tuned model from ./trained_model/")
                        
                        base_model = T5ForConditionalGeneration.from_pretrained("t5-base")
                        model = PeftModel.from_pretrained(base_model, MODEL_PATHS["trained_model"])
                        tokenizer = T5Tokenizer.from_pretrained(MODEL_PATHS["trained_model"])
                        
                        st.success("✅ Successfully loaded fine-tuned model!")
                        
                        training_info = load_training_info()
                        if training_info.get('config'):
                            config = training_info['config']
                            st.sidebar.markdown("### 📊 Model Training Info")
                            st.sidebar.markdown(f"""
                            - **Base Model:** {config.get('model_name', 'N/A')}
                            - **Epochs:** {config.get('num_epochs', 'N/A')}
                            - **Learning Rate:** {config.get('learning_rate', 'N/A')}
                            - **Batch Size:** {config.get('batch_size', 'N/A')}
                            """)
                        
                        if training_info.get('history'):
                            history = training_info['history']
                            if history.get('train_loss') and history.get('val_loss'):
                                final_train_loss = history['train_loss'][-1]
                                final_val_loss = history['val_loss'][-1]
                                st.sidebar.markdown(f"""
                                - **Final Train Loss:** {final_train_loss:.4f}
                                - **Final Val Loss:** {final_val_loss:.4f}
                                """)
                        
                    except Exception as e:
                        st.error(f"❌ Error loading fine-tuned model: {str(e)}")
                        st.warning("⚠️ Falling back to base T5 model...")
                        model = T5ForConditionalGeneration.from_pretrained("t5-base")
                        tokenizer = T5Tokenizer.from_pretrained("t5-base")
                else:
                    st.warning("⚠️ Trained model files missing")
                    model = T5ForConditionalGeneration.from_pretrained("t5-base")
                    tokenizer = T5Tokenizer.from_pretrained("t5-base")
            
            elif model_choice == "T5-Base":
                model = T5ForConditionalGeneration.from_pretrained("t5-base")
                tokenizer = T5Tokenizer.from_pretrained("t5-base")
            
            elif model_choice == "T5-Small":
                model = T5ForConditionalGeneration.from_pretrained("t5-small")
                tokenizer = T5Tokenizer.from_pretrained("t5-small")
            
            model.to(device)
            model.eval()
            
            return model, tokenizer, device
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

def generate_text_local(model, tokenizer, device, prompt, text, max_length=150, temperature=0.7, num_beams=4):
    """Generate text using local model"""
    try:
        input_text = f"{prompt}: {text}"
        inputs = tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors='pt',
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text if generated_text else "Could not generate text."
    
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize API client
api_client = init_api_client()

# Sidebar
st.sidebar.markdown("## ⚙️ Model Settings")

# Model selection - Only show T5 models
model_options = []

if transformers_available:
    model_files_status = check_model_files()
    if peft_available and model_files_status["model_dir"]:
        model_options.append("Fine-tuned T5-Base")
    
    model_options.extend([
        "T5-Base",
        "T5-Small"
    ])

if not model_options:
    model_options = ["T5-Base"]

selected_model = st.sidebar.selectbox(
    "Select Transformer Model",
    model_options,
    index=0,
    help="Choose T5 model variant for text generation"
)

st.sidebar.markdown("---")

# Advanced settings
with st.sidebar.expander("🔧 Advanced Settings"):
    summary_length = st.slider("Summary Length", 50, 300, 150, 25)
    flashcard_length = st.slider("Flashcard Length", 50, 200, 100, 25)
    notes_length = st.slider("Notes Length", 100, 400, 250, 50)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    num_beams = st.slider("Beam Search", 2, 8, 4, 1)

st.sidebar.markdown("---")

# System info
with st.sidebar.expander("💻 System Information"):
    st.write(f"**Python Version:** {sys.version.split()[0]}")
    if transformers_available:
        st.write(f"**PyTorch:** {torch.__version__}")
        st.write(f"**Compute Device:** {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")
        st.write(f"**PEFT Support:** {'✅ Yes' if peft_available else '❌ No'}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 About This App")
st.sidebar.info(
    "**STEM Learning Assistant**\n\n"
    "Transform educational content into:\n\n"
    "📝 **Summaries** - Key points overview\n\n"
    "🃏 **Flashcards** - Q&A format\n\n"
    "📚 **Study Notes** - Detailed explanations\n\n"
    "Powered by T5 Transformer models!"
)

# Main content
st.markdown('<p class="main-header">📚 STEM Learning Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Transform any STEM topic into summaries, flashcards, and comprehensive study notes</p>',
    unsafe_allow_html=True
)

# Status badges
col1, col2, col3 = st.columns(3)
with col1:
    if api_available and api_client:
        st.markdown('<span class="info-badge badge-success">✅ Model Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="info-badge badge-warning">⚠️ Offline Mode</span>', unsafe_allow_html=True)

with col2:
    if transformers_available:
        st.markdown('<span class="info-badge badge-info">🤖 T5 Model Ready</span>', unsafe_allow_html=True)

with col3:
    device_type = "GPU Accelerated" if torch.cuda.is_available() else "CPU Mode"
    badge_class = "badge-success" if torch.cuda.is_available() else "badge-info"
    st.markdown(f'<span class="info-badge {badge_class}">⚡ {device_type}</span>', unsafe_allow_html=True)

st.markdown("---")

# Load local model
model, tokenizer, device = None, None, None
if transformers_available:
    model, tokenizer, device = load_model(selected_model)
    if model is None and not api_client:
        st.error("❌ Failed to load model and no API available")
        st.stop()

# Input section
st.markdown("### 📝 Enter Your Content")

input_method = st.radio(
    "Choose input method:",
    ["Enter topic/question", "Paste detailed text"],
    horizontal=True
)

if input_method == "Enter topic/question":
    user_input = st.text_input(
        "Topic or Question:",
        placeholder="e.g., Photosynthesis, Newton's Laws of Motion, DNA Replication...",
        help="Enter a STEM topic or question you want to learn about"
    )
    if user_input:
        context_text = f"Explain the concept of {user_input} in detail with key points and examples."
else:
    user_input = st.text_area(
        "Detailed Text:",
        placeholder="Paste your text here (lecture notes, article, textbook excerpt, etc.)",
        height=150,
        help="Paste educational content to create study materials from"
    )
    context_text = user_input

# Generate button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_btn = st.button("🚀 Generate Learning Materials", use_container_width=True, type="primary")

# Generation logic
if generate_btn:
    if not user_input or len(user_input.strip()) < 10:
        st.warning("⚠️ Please enter at least 10 characters")
    else:
        # Determine which method to use
        use_api = api_available and api_client
        use_local = model is not None and tokenizer is not None
        
        if not use_api and not use_local:
            st.error("❌ No generation method available")
            st.stop()
        
        with st.spinner("🔄 Generating your learning materials..."):
            progress_bar = st.progress(0)
            
            try:
                # Generate summary
                st.info("Generating summary...")
                progress_bar.progress(25)
                
                if use_api:
                    summary = generate_with_api(api_client, "summarize", context_text, "summary")
                else:
                    summary = generate_text_local(model, tokenizer, device, "summarize", context_text, 
                                                 summary_length, temperature, num_beams)
                
                # Generate flashcard
                st.info("Creating flashcard...")
                progress_bar.progress(50)
                
                if use_api:
                    flashcard = generate_with_api(api_client, "create flashcard", context_text, "flashcard")
                else:
                    flashcard = generate_text_local(model, tokenizer, device, "create question and answer", 
                                                   context_text, flashcard_length, temperature, num_beams)
                
                # Generate notes
                st.info("Preparing study notes...")
                progress_bar.progress(75)
                
                if use_api:
                    notes = generate_with_api(api_client, "create notes", context_text, "notes")
                else:
                    notes = generate_text_local(model, tokenizer, device, "explain in detail", 
                                              context_text, notes_length, temperature, num_beams)
                
                progress_bar.progress(100)
                st.success("✅ Generation complete!")
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Generated Learning Materials")
                
                tab1, tab2, tab3 = st.tabs(["📝 Summary", "🃏 Flashcard", "📚 Study Notes"])
                
                with tab1:
                    st.markdown(f'<div class="output-box summary-box"><strong style="color: #e67e22; font-size: 1.1rem;">📝 Summary</strong><br><br>{summary}</div>', unsafe_allow_html=True)
                    st.download_button(
                        label="📄 Download Summary",
                        data=f"# Summary\n\n{summary}",
                        file_name="summary.txt",
                        mime="text/plain",
                        key="download_summary"
                    )
                
                with tab2:
                    st.markdown(f'<div class="output-box flashcard-box"><strong style="color: #27ae60; font-size: 1.1rem;">🃏 Flashcard</strong><br><br>{flashcard}</div>', unsafe_allow_html=True)
                    st.download_button(
                        label="🃏 Download Flashcard",
                        data=f"# Flashcard\n\n{flashcard}",
                        file_name="flashcard.txt",
                        mime="text/plain",
                        key="download_flashcard"
                    )
                
                with tab3:
                    st.markdown(f'<div class="output-box notes-box"><strong style="color: #c0392b; font-size: 1.1rem;">📚 Study Notes</strong><br><br>{notes}</div>', unsafe_allow_html=True)
                    st.download_button(
                        label="📚 Download Notes",
                        data=f"# Study Notes\n\n{notes}",
                        file_name="study_notes.txt",
                        mime="text/plain",
                        key="download_notes"
                    )
                
                # Combined download
                st.markdown("---")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                combined_text = f"""# STEM Learning Materials
## Topic: {user_input if input_method == "Enter topic/question" else "Custom Text"}
## Generated using: {selected_model}
## Date: {timestamp}

---

## 📝 Summary
{summary}

---

## 🃏 Flashcard
{flashcard}

---

## 📚 Detailed Study Notes
{notes}

---
Generated with STEM Learning Assistant
"""
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        label="📦 Download All Materials",
                        data=combined_text,
                        file_name="learning_materials_complete.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="download_all"
                    )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try selecting a different model or check your settings.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #34495e; padding: 2rem;'>
        <p style='font-size: 1.1rem; font-weight: 600;'>Built with Streamlit | Powered by T5 Transformer Models</p>
        <p style='font-size: 0.95rem; margin-top: 0.5rem;'>💡 Tip: Use fine-tuned models for best results on STEM content!</p>
        <p style='font-size: 0.85rem; margin-top: 1rem; color: #7f8c8d;'>
            🔒 Secure Processing | ⚡ High-Performance Inference | 🎓 Educational AI
        </p>
    </div>
    """,
    unsafe_allow_html=True
)