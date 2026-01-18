# 🚀 Quick Start Guide - Emotion Recognition Dashboard

## ⏱️ 5 Minutes to Get Started

### Step 1: Prepare the Model (1 minute)

```bash
# Download from Google Drive:
# /content/drive/MyDrive/blip2-emotion-rafce-final

# Extract to project folder:
# FER_AI_Project/blip2-emotion-rafce-final/
```

### Step 2: Install Dependencies (2 minutes)

```bash
# Navigate to project
cd c:\Users\famil\Desktop\ghaith\Projects\FER_AI_Project

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install packages
pip install -r requirements_streamlit.txt
```

### Step 3: Run Dashboard (2 minutes)

```bash
streamlit run streamlit_app.py
```

**Done!** 🎉 Dashboard opens at `http://localhost:8501`

---

## 📖 Detailed Setup

### Prerequisites Check

```bash
python setup_dashboard.py --setup
```

This script will:

- ✅ Check Python version
- ✅ Detect GPU availability
- ✅ Verify disk space
- ✅ Install missing packages
- ✅ Validate model files

### Troubleshooting Quick Fixes

**Issue**: `ModuleNotFoundError: No module named 'streamlit'`

```bash
pip install -r requirements_streamlit.txt
```

**Issue**: `CUDA out of memory`

- Reduce model token generation: Edit `streamlit_app.py` line 195
- Change `max_new_tokens=200` to `max_new_tokens=100`

**Issue**: `No face detected`

- Ensure clear, well-lit facial images
- Face should occupy 30-80% of image
- Try a different image

---

## 🎯 Using the Dashboard

### Upload & Analyze Tab

1. Click "Choose an image file"
2. Select a facial image
3. Review the aligned face
4. Click "🔍 Analyze Emotion"
5. View results

### Batch Processing Tab

1. Select multiple images (Ctrl+Click)
2. Click "🚀 Process Batch"
3. Wait for completion
4. Click "📥 Download Results (CSV)"

### Output Format

```
This face exhibits: Happiness, Surprise.
Emotion vector: [0, 0, 0, 1, 0, 0].
Observed Action Units: 1+4+12+25
```

---

## 🔧 Common Customizations

### Change Model Path

Edit `streamlit_app.py` line ~70:

```python
model_path = st.sidebar.text_input(
    "Model Path (LoRA adapters)",
    value="./custom_path/to/model"  # Change here
)
```

### Adjust Inference Parameters

Edit `streamlit_app.py` line ~195:

```python
generated_ids = model.generate(
    **inputs,
    max_new_tokens=200,      # Shorter = faster
    do_sample=True,
    temperature=0.7,         # Lower = more focused
    top_p=0.9,
)
```

### Reduce Memory Usage

Edit `streamlit_app.py` line ~170:

```python
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    device_map="auto",
    torch_dtype=torch.float16  # Change to float32 if memory issues
)
```

---

## 📊 Performance Tips

| Setting          | Fast | Accurate | Recommended |
| ---------------- | ---- | -------- | ----------- |
| `max_new_tokens` | 50   | 200      | 150         |
| `temperature`    | 0.3  | 0.9      | 0.7         |
| `batch_size`     | 1    | 8        | 4           |
| `top_p`          | 0.5  | 1.0      | 0.9         |

---

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All packages installed: `pip install -r requirements_streamlit.txt`
- [ ] Model directory exists: `./blip2-emotion-rafce-final/`
- [ ] Can run: `streamlit run streamlit_app.py`
- [ ] Dashboard loads at `http://localhost:8501`
- [ ] Can upload image without errors
- [ ] Face detection works
- [ ] Model inference completes
- [ ] Output is generated

---

## 🎓 Understanding the Output

### Emotion Classes

```
Surprise  😮 - Unexpected reaction
Fear      😨 - Anxiety or dread
Disgust   😖 - Dislike or revulsion
Happiness 😊 - Joy and contentment
Sadness   😢 - Sorrow or grief
Anger     😠 - Displeasure or rage
```

### Emotion Vector

```
[0, 0, 1, 0, 0, 0]
 ↓  ↓  ↓  ↓  ↓  ↓
 S  F  D  H  Sa A

0 = not present, 1 = present
```

### Action Units

```
1  = Inner Brow Raiser
4  = Brow Lowerer
12 = Lip Corner Puller (smile)
25 = Lips part
```

---

## 🎥 Example Usage

### Single Image

1. Upload: `happy_face.jpg`
2. Wait for processing
3. See: Aligned face + emotion analysis

### Multiple Images (Batch)

1. Select 10 images
2. Click "Process Batch"
3. Watch progress
4. Export CSV with results

---

## 📞 Need Help?

1. **Check logs**: Look at terminal output
2. **Read full guide**: Open `STREAMLIT_README.md`
3. **Run verification**: `python setup_dashboard.py`
4. **Test model**: `python setup_dashboard.py --test-inference`

---

## 🚫 Common Errors & Fixes

```
❌ Error: CUDA out of memory
✅ Fix: pip install --upgrade torch
     or change max_new_tokens to 100

❌ Error: No module named 'peft'
✅ Fix: pip install peft

❌ Error: Image shape mismatch
✅ Fix: Ensure image format is RGB/RGBA
     or change cv2.COLOR_* constants

❌ Error: Connection refused localhost:8501
✅ Fix: streamlit run streamlit_app.py
```

---

## 📈 Next Steps After Setup

1. **Test with sample images**
   - Try different face angles
   - Test batch processing
   - Check CSV export

2. **Fine-tune parameters**
   - Adjust temperature for consistency
   - Modify token length for brevity
   - Test batch size for speed

3. **Integrate into workflow**
   - API wrapper (Flask/FastAPI)
   - Database storage
   - Web deployment

---

## 🎯 Final Checklist

```bash
# Terminal commands
cd c:\Users\famil\Desktop\ghaith\Projects\FER_AI_Project
venv\Scripts\activate
pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
```

Open browser → `http://localhost:8501` → ✅ Done!

---

**Status**: Ready to Use  
**Support**: See STREAMLIT_README.md  
**Version**: 1.0
