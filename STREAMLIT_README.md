# 🎭 Facial Emotion Recognition Dashboard

A Streamlit-based interactive dashboard for emotion recognition using fine-tuned BLIP-2 model with facial Action Unit analysis.

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **GPU** (Recommended): NVIDIA GPU with CUDA support (10GB+ VRAM)
- **Storage**: ~6GB for model files
- **OS**: Windows, macOS, or Linux

## 🚀 Setup Instructions

### Step 1: Download Fine-tuned Model

First, download the fine-tuned model parameters from Google Drive:

```
/content/drive/MyDrive/blip2-emotion-rafce-final
```

Extract the files to your project directory:

```
FER_AI_Project/
├── blip2-emotion-rafce-final/     # ← Fine-tuned model directory
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── config.json
├── streamlit_app.py
└── requirements_streamlit.txt
```

### Step 2: Create Virtual Environment

```bash
# Navigate to project directory
cd c:\Users\famil\Desktop\ghaith\Projects\FER_AI_Project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements_streamlit.txt

# Verify PyTorch installation (especially GPU support)
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Step 4: Download Base Model (Optional - Auto-downloads on first run)

The application will automatically download the base BLIP-2 model on first run:

```bash
python -c "from transformers import Blip2Processor, Blip2ForConditionalGeneration; Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b'); Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')"
```

### Step 5: Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📂 Project Structure

```
FER_AI_Project/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_blip_training.ipynb
│   └── Final_notebook.ipynb          # Source implementation
├── config/
│   ├── mlops_config.yaml
│   └── requirements.txt
├── blip2-emotion-rafce-final/        # Fine-tuned LoRA adapters
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── config.json
├── streamlit_app.py                  # Main Streamlit application
├── requirements_streamlit.txt         # Python dependencies
└── README.md                          # This file
```

## 🎯 Features

### 1. **Single Image Analysis** (Tab 1)

- Upload a facial image
- Automatic face detection and alignment
- Real-time emotion analysis
- Multi-label emotion support
- Action Unit (AU) identification

### 2. **Batch Processing** (Tab 2)

- Process multiple images simultaneously
- Bulk emotion analysis
- CSV export of results
- Progress tracking
- Error handling per image

### 3. **About & Documentation** (Tab 3)

- Model architecture details
- Feature explanations
- Best practices for accuracy
- Technical references

## 💡 Usage Guide

### Basic Analysis

1. **Open the Dashboard**

   ```bash
   streamlit run streamlit_app.py
   ```

2. **Select "Upload & Analyze" Tab**

3. **Upload an Image**
   - Click "Choose an image file"
   - Select a facial image (JPG, PNG, BMP)
   - Supported formats: JPEG, PNG, BMP

4. **Review Preprocessed Face**
   - See the detected and aligned face
   - Verify face detection was successful

5. **Click "Analyze Emotion"**
   - Wait for model inference (5-10 seconds on GPU)
   - Review the emotion analysis output

### Batch Processing

1. **Select "Batch Processing" Tab**

2. **Upload Multiple Images**
   - Select multiple images using Ctrl+Click or Shift+Click
   - Up to 100+ images supported (depends on VRAM)

3. **Click "Process Batch"**
   - Monitor progress with status indicators
   - Results appear in real-time

4. **Download Results**
   - Click "Download Results (CSV)"
   - CSV contains: Filename, Status, Prediction

## 📊 Model Output Format

The model generates text in this format:

```
This face exhibits: [Emotions].
Emotion vector: [6D vector].
Observed Action Units: [AU string]
Explanation: [Connection between AUs and emotions]
```

### Example Output

```
This face exhibits: Happiness, Surprise.
Emotion vector: [0, 0, 0, 1, 0, 0].
Observed Action Units: 1+4+12+25
Explanation: The wide smile (AU 12) combined with raised eyebrows (AU 1 and 4) and open mouth (AU 25) clearly indicates happiness and surprise.
```

## ⚙️ Configuration

### Model Path

The default model path is `./blip2-emotion-rafce-final`. You can change this in the sidebar:

```python
model_path = "./custom_path/to/model"
```

### Inference Parameters

Modify in `streamlit_app.py` (line ~200):

```python
generated_ids = model.generate(
    **inputs,
    max_new_tokens=200,      # Increase for longer responses
    do_sample=True,
    temperature=0.7,         # Lower = more deterministic
    top_p=0.9,              # Nucleus sampling
)
```

### Device Selection

Auto-detects GPU, set manually (if needed):

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"

**Solution**:

- Reduce batch size (in app, process fewer images at once)
- Use CPU mode: Set `device = "cpu"` in code
- Reduce `max_new_tokens` parameter

### Issue: "Module not found"

**Solution**:

```bash
pip install --upgrade -r requirements_streamlit.txt
```

### Issue: "Face not detected"

**Solution**:

- Ensure face is clearly visible and well-lit
- Face should occupy ~20-80% of image
- Try front-facing or slightly angled images
- Avoid sunglasses or extreme shadows

### Issue: "Model loading failed"

**Solution**:

- Verify LoRA adapters exist in `blip2-emotion-rafce-final/`
- Check internet connection (for base model download)
- Ensure sufficient disk space (~6GB)
- Try: `pip install --upgrade transformers peft`

### Issue: "Slow inference (>30 seconds)"

**Solution**:

- Verify GPU is being used: Check sidebar indicator
- Reduce `max_new_tokens` (default: 200 → try 100)
- Close other GPU applications
- Use smaller batch in batch processing

## 📈 Performance Tips

### For Better Accuracy

1. **Image Quality**: Use clear, well-lit images
2. **Face Size**: Face should be 100-500px wide
3. **Angle**: Front-facing or ±45° optimal
4. **Single Face**: Process one face per image

### For Speed

1. **GPU Mode**: Ensure CUDA is available
2. **Batch Size**: Process 4-8 images together
3. **Token Limit**: Reduce `max_new_tokens` to 100
4. **Temperature**: Lower temperature = faster inference

## 📚 Model Information

- **Architecture**: BLIP-2 OPT 2.7B + LoRA
- **Base Model**: Salesforce/blip2-opt-2.7b
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Data**: RAF-ML + RAF-AU
- **Emotions**: 6 classes (Surprise, Fear, Disgust, Happiness, Sadness, Anger)
- **Multi-label**: Yes (supports multiple emotions per image)
- **Input Size**: 336×336 pixels
- **Output**: Natural language description with emotions + AUs

## 🔧 Advanced Usage

### Load Custom Model Path

```python
# In Python terminal or script
import streamlit as st
st.set_page_config(...)

# Modify model path before running
model_path = "/path/to/custom/model"
```

### Export Detailed Results

The app automatically exports CSV. For JSON export, modify the batch processing tab:

```python
import json

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Integrate with Other Systems

The `analyze_emotion()` function is standalone and can be used in other scripts:

```python
from streamlit_app import analyze_emotion, FaceAlignmentPreprocessor
from PIL import Image

# Load image
image = Image.open("face.jpg")

# Analyze
result = analyze_emotion(model, processor, image, device)
print(result)
```

## 📞 Support

For issues or questions:

1. **Check logs**: Look at terminal output
2. **Verify setup**: Run through Step 2-3 again
3. **Test model**: Run test_model.py (see below)

### Test Model Independently

```python
# test_model.py
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel

# Test base model load
print("Testing base model...")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
print("✅ Base model loaded")

# Test LoRA adapters
print("Testing LoRA adapters...")
model = PeftModel.from_pretrained(model, "./blip2-emotion-rafce-final")
print("✅ LoRA adapters loaded")

print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
print("✅ All checks passed!")
```

Run with:

```bash
python test_model.py
```

## 📖 References

- [BLIP-2 Paper](https://arxiv.org/abs/2301.12597)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Facial Action Units](https://en.wikipedia.org/wiki/Facial_action_coding_system)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

## 📝 License

This project uses:

- BLIP-2: Apache 2.0 License (Salesforce)
- LoRA: MIT License
- Streamlit: Apache 2.0 License
- OpenCV: Apache 2.0 License

## ✅ Checklist for First Run

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements_streamlit.txt`)
- [ ] Fine-tuned model downloaded to `blip2-emotion-rafce-final/`
- [ ] CUDA verified if using GPU
- [ ] Streamlit app runs: `streamlit run streamlit_app.py`
- [ ] Can upload and process test image
- [ ] Output matches expected format

## 🎯 Next Steps

1. **Test with Sample Images**: Use diverse face images
2. **Fine-tune Settings**: Adjust temperature, token length
3. **Integrate**: Add to production pipeline if needed
4. **Monitor**: Track inference times and accuracy

---

**Version**: 1.0  
**Last Updated**: January 2026  
**Status**: ✅ Production Ready
