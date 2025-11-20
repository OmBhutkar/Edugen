# 🎓 EduGen - The Learning Assistant

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**EduGen** is an advanced AI-powered educational platform that leverages four state-of-the-art deep learning models to transform educational content creation, summarization, and visualization. Built with Streamlit, it provides an intuitive interface for students, educators, and content creators.

## 🌟 Features

### 🎯 GAN - Question Generation
- **Intelligent Question Creation**: Generates diverse educational questions from any text content
- **Seq2Seq Architecture**: Uses advanced Sequence-to-Sequence GAN with Attention and Coverage mechanisms
- **Customizable Output**: Adjust the number of questions (1-15) based on your needs
- **Multi-level Questions**: Creates factual, conceptual, and application-based questions

### 🎨 VAE - Diagram Compression
- **Efficient Compression**: Compress educational diagrams while maintaining visual quality
- **Variational Autoencoder**: Leverages VAE technology for intelligent image reconstruction
- **AI-Powered Analysis**: Provides detailed analysis of compressed images including:
  - Content Analysis
  - Visual Quality Assessment
  - Educational Value Evaluation
  - Compression Trade-offs
  - Usage Recommendations
- **Adjustable Compression**: 10 compression levels (1-10) for optimal size-quality balance
- **Download Options**: Save compressed images and analysis reports

### 📝 Transformer - Summarization & Notes
- **Dual Mode Processing**: 
  - **Summary Mode**: Creates concise, comprehensive summaries
  - **Study Notes Mode**: Generates detailed, well-organized study materials
- **Advanced NLP**: Uses state-of-the-art Transformer architecture
- **Structured Output**: Includes key concepts, explanations, examples, and important points
- **Downloadable Results**: Export summaries and notes in text format

### 🖼️ Diffusion - Text-to-Illustration
- **AI-Generated Diagrams**: Creates stunning, realistic educational illustrations from text descriptions
- **SVG Format**: Scalable Vector Graphics for high-quality, resolution-independent output
- **Scientific Accuracy**: Generates scientifically accurate diagrams with proper labeling
- **Detailed Explanations**: Optional comprehensive educational explanations including:
  - Overview
  - Key Components
  - How It Works
  - Educational Significance
  - Real-World Applications
  - Interesting Facts
- **Multiple Styles**: Realistic scientific visualization
- **Example Topics**: Biology, Physics, Chemistry, Mathematics diagrams

## 🚀 Live Demo

[**Try EduGen Live**](https://your-app-url.streamlit.app) *(Update with your actual deployment URL)*

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)


## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/edugen.git
cd edugen
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Download Pre-trained Models
If you have trained models, place them in the `trained_model/` directory:
```
trained_model/
├── seq2seq_attn_cov.pt      # GAN model weights
├── vae_model.h5              # VAE model
├── encoder.h5                # VAE encoder
├── decoder.h5                # VAE decoder
├── adapter_model.safetensors # Transformer model
└── config.json               # Model configuration
```

## 🎮 Usage

### Run Locally
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Models

#### 1. GAN - Question Generation
1. Select "🎯 GAN - Question Generation" from the sidebar
2. Enter or paste educational content (minimum 20 characters)
3. Adjust the number of questions using the slider (1-15)
4. Click "🚀 Generate Questions"
5. Download the generated questions as a text file

#### 2. VAE - Diagram Compression
1. Select "🎨 VAE - Diagram Compression" from the sidebar
2. Upload an image file (PNG, JPG, JPEG, BMP)
3. Adjust the compression level (1-10)
4. Click "🚀 Process with VAE"
5. View compression statistics and AI analysis
6. Download compressed image and analysis report

#### 3. Transformer - Summarization/Notes
1. Select "📝 Transformer - Summarization/Notes" from the sidebar
2. Choose output type (Summary or Detailed Study Notes)
3. Enter educational content (minimum 20 characters)
4. Click "🚀 Generate"
5. Download the result as a text file

#### 4. Diffusion - Text-to-Illustration
1. Select "🖼️ Diffusion - Text-to-Illustration" from the sidebar
2. Enter a detailed description of the desired illustration
3. Select art style and quality settings
4. Optionally enable detailed explanation
5. Click "🚀 Generate Illustration"
6. View, download SVG, code, or complete package

## 🏗️ Architecture

### Technology Stack
- **Frontend**: Streamlit
- **Backend**: Groq API (LLaMA 3.3 70B)
- **Deep Learning Frameworks**:
  - PyTorch (GAN model)
  - TensorFlow/Keras (VAE model)
  - Transformers (NLP models)
- **Image Processing**: Pillow (PIL)
- **Scientific Computing**: NumPy

### Model Details

#### GAN (Generative Adversarial Network)
- **Architecture**: Sequence-to-Sequence with Attention and Coverage
- **Components**: 
  - Bidirectional LSTM Encoder
  - LSTM Decoder with Attention mechanism
  - Coverage mechanism to prevent repetition
- **Vocab Size**: 8,004 tokens
- **Embedding Dimension**: 200
- **Hidden Dimension**: 256

#### VAE (Variational Autoencoder)
- **Purpose**: Image compression and reconstruction
- **Architecture**: Encoder-Decoder structure
- **Compression**: JPEG-based with quality adjustment
- **Output**: Reconstructed images with compression statistics

#### Transformer
- **Model**: LLaMA 3.3 70B (via Groq API)
- **Tasks**: Text summarization, note generation
- **Max Tokens**: 2,048
- **Temperature**: 0.7 (balanced creativity)

#### Diffusion
- **Model**: LLaMA 3.3 70B (via Groq API)
- **Output**: SVG vector graphics
- **Features**: Gradients, shadows, scientific accuracy
- **Max Tokens**: 6,000

## 📊 Project Structure

```
edugen/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .env                      # Environment variables (optional)
├── .gitignore               # Git ignore file
├── trained_model/           # Pre-trained model weights (optional)
│   ├── seq2seq_attn_cov.pt
│   ├── vae_model.h5
│   ├── encoder.h5
│   ├── decoder.h5
│   ├── adapter_model.safetensors
│   └── config.json
└── assets/                  # Images and resources (optional)
```

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file as `app.py`
   - Click "Deploy"

3. **Add Secrets** (in Streamlit Cloud dashboard):
   ```toml
   # .streamlit/secrets.toml
   GROQ_API_KEY = "your-api-key-here"
   ```

### Deploy to Other Platforms

#### Heroku
```bash
heroku create your-app-name
git push heroku main
heroku config:set GROQ_API_KEY=your-api-key
```

#### Railway
1. Connect GitHub repository
2. Add environment variable `GROQ_API_KEY`
3. Deploy automatically


## 🎨 Customization

### Modify Compression Levels
Edit `app.py` line ~730:
```python
quality = max(10, 100 - (compression_level * 8))  # Adjust formula
```

### Change Question Count Range
Edit `app.py` line ~640:
```python
num_questions = st.slider(
    "Number of Questions to Generate",
    min_value=1,
    max_value=15,  # Change max value
    value=5,
    step=1
)
```

### Customize UI Colors
Edit CSS in `app.py` (lines 66-254) to match your brand:
```python
.main-header {
    background: linear-gradient(135deg, #your-color1, #your-color2);
}
```

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

**2. API Rate Limit Exceeded**
- Wait 24 hours for token reset
- Use a different Groq API key
- Reduce token usage by generating fewer/shorter outputs

**3. Model Loading Fails**
- Ensure model files are in `trained_model/` directory
- Check file paths in `app.py` (lines 47-55)
- Models are optional; API backend will be used as fallback

**4. Image Upload Issues**
- Supported formats: PNG, JPG, JPEG, BMP
- Check file size (recommended < 10MB)
- Ensure image is not corrupted

## 📈 Performance Optimization

### For Faster Processing
1. Use GPU if available (PyTorch/TensorFlow will auto-detect)
2. Reduce compression quality for faster VAE processing
3. Limit token count in API calls
4. Cache results using `@st.cache_data`

### Memory Management
```python
# Clear cache manually
st.cache_data.clear()
st.cache_resource.clear()
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** - For the amazing web framework
- **Groq** - For providing fast AI inference
- **PyTorch & TensorFlow** - For deep learning capabilities
- **Hugging Face** - For transformer models and tools

## 📧 Contact

- **Author**: Sahil Karne (sahil.karne@mitaoe.ac.in) , Sachin Jadhav (sachin.jadhav@mitaoe.ac.in) ,Yash Gunjal (yash.gunjal@mitaoe.ac.in) ,Om Bhutkar (om.bhutkar@mitaoe.ac.in) ,Aryan Tamboli (aryan.tamboli@mitaoe.ac.in)
- **PRN**:202201040086,202201040080,202201040106,202201040111,202201040088

## 🔮 Future Enhancements

- [ ] Add more AI models (RL-based learning paths)
- [ ] Multi-language support
- [ ] Collaborative features (shared workspaces)
- [ ] Mobile app version
- [ ] Integration with LMS platforms
- [ ] Real-time collaboration
- [ ] Advanced analytics dashboard
- [ ] Custom model training interface
- [ ] API endpoints for external integration
- [ ] Voice input/output capabilities

## 📊 Statistics

- **Models**: 4 AI models (GAN, VAE, Transformer, Diffusion)
- **Languages**: Python, HTML, CSS
- **Lines of Code**: ~1,200+
- **Dependencies**: 20+ packages
- **Features**: 15+ major features

---

<div align="center">

**Built with Pride for Better Education**

⭐ Star this repo if you find it helpful!

[Report Bug](https://github.com/yourusername/edugen/issues) • [Request Feature](https://github.com/yourusername/edugen/issues)

</div>
