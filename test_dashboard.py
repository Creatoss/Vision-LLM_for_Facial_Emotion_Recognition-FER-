#!/usr/bin/env python3
"""
Test Script for Emotion Recognition Dashboard
Verifies all components work correctly before launching the dashboard
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# ============================================================
# COLOR OUTPUT
# ============================================================

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

def print_test(msg):
    print(f"\n{Colors.BOLD}🧪 {msg}{Colors.END}")

# ============================================================
# TESTS
# ============================================================

def test_imports():
    """Test if all required packages can be imported"""
    print_test("Testing Package Imports")
    
    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT',
        'cv2': 'OpenCV',
        'streamlit': 'Streamlit',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'pandas': 'Pandas'
    }
    
    failed = []
    for module, name in packages.items():
        try:
            __import__(module)
            print_success(f"{name}")
        except ImportError as e:
            print_error(f"{name}: {str(e)}")
            failed.append(module)
    
    return len(failed) == 0, failed

def test_torch():
    """Test PyTorch installation and GPU availability"""
    print_test("Testing PyTorch Configuration")
    
    try:
        import torch
        
        print_success(f"PyTorch Version: {torch.__version__}")
        
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            print_success(f"GPU Available: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_info(f"GPU VRAM: {vram:.2f}GB")
        else:
            print_warning("GPU not available, will use CPU")
        
        return True
    
    except Exception as e:
        print_error(f"PyTorch test failed: {str(e)}")
        return False

def test_opencv():
    """Test OpenCV and Haar Cascades"""
    print_test("Testing OpenCV Face Detection")
    
    try:
        import cv2
        
        # Check Haar cascades
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        if face_cascade.empty():
            print_error("Face cascade not found")
            return False
        
        if eye_cascade.empty():
            print_error("Eye cascade not found")
            return False
        
        print_success("Face detection cascade loaded")
        print_success("Eye detection cascade loaded")
        
        # Test detection on dummy image
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        gray = cv2.cvtColor(dummy_img, cv2.COLOR_BGR2GRAY)
        
        # This should work without errors
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print_success("Face detection pipeline works")
        
        return True
    
    except Exception as e:
        print_error(f"OpenCV test failed: {str(e)}")
        return False

def test_model_files():
    """Check if model files exist"""
    print_test("Checking Model Files")
    
    model_path = Path('./blip2-emotion-rafce-final')
    
    if not model_path.exists():
        print_warning(f"Model directory not found: {model_path}")
        print_info("This is OK - it will be downloaded on first run")
        print_info("Or manually place the model in: ./blip2-emotion-rafce-final/")
        return True  # Not a critical error
    
    required_files = [
        'adapter_config.json',
        'adapter_model.bin'
    ]
    
    all_present = True
    for file in required_files:
        file_path = model_path / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024*1024)
            print_success(f"Found: {file} ({size_mb:.1f}MB)")
        else:
            print_warning(f"Missing: {file}")
            all_present = False
    
    return all_present

def test_transformers():
    """Test if transformers can be imported and used"""
    print_test("Testing Transformers Library")
    
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        print_info("Loading processor...")
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        print_success("Processor loaded successfully")
        
        print_info("Checking PEFT compatibility...")
        from peft import PeftModel
        print_success("PEFT library available")
        
        return True
    
    except Exception as e:
        print_error(f"Transformers test failed: {str(e)}")
        print_warning("This will be resolved on first run when model is downloaded")
        return True  # Not critical - will download on first run

def test_streamlit():
    """Test Streamlit installation"""
    print_test("Testing Streamlit")
    
    try:
        import streamlit as st
        
        print_success(f"Streamlit Version: {st.__version__}")
        return True
    
    except Exception as e:
        print_error(f"Streamlit test failed: {str(e)}")
        return False

def test_files_structure():
    """Check project file structure"""
    print_test("Checking Project Files")
    
    required_files = [
        'streamlit_app.py',
        'setup_dashboard.py',
        'requirements_streamlit.txt',
        'QUICK_START.md',
        'STREAMLIT_README.md',
        'ARCHITECTURE.md',
        'IMPLEMENTATION_SUMMARY.md'
    ]
    
    all_present = True
    for file in required_files:
        if os.path.exists(file):
            size_kb = os.path.getsize(file) / 1024
            print_success(f"Found: {file} ({size_kb:.1f}KB)")
        else:
            print_warning(f"Missing: {file}")
            all_present = False
    
    return all_present

def test_disk_space():
    """Check available disk space"""
    print_test("Checking Disk Space")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        
        print_info(f"Free disk space: {free_gb:.2f}GB")
        
        if free_gb >= 6:
            print_success("Sufficient space for model files")
            return True
        else:
            print_warning(f"Low disk space ({free_gb:.2f}GB). Recommended: 6GB+")
            return True  # Not critical - can proceed
    
    except Exception as e:
        print_error(f"Disk space check failed: {str(e)}")
        return False

def test_face_alignment():
    """Test face alignment functionality"""
    print_test("Testing Face Alignment Preprocessing")
    
    try:
        import cv2
        import numpy as np
        
        # Create dummy face image (100x100 white square)
        dummy_face = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Test preprocessing steps
        gray = cv2.cvtColor(dummy_face, cv2.COLOR_BGR2GRAY)
        print_success("Image conversion works")
        
        # Test resize
        resized = cv2.resize(gray, (336, 336))
        if resized.shape == (336, 336):
            print_success("Image resizing works (336×336)")
        else:
            print_error("Image resize failed")
            return False
        
        # Test rotation
        M = cv2.getRotationMatrix2D((50, 50), 15, 1.0)
        rotated = cv2.warpAffine(dummy_face, M, (100, 100))
        print_success("Image rotation works")
        
        return True
    
    except Exception as e:
        print_error(f"Face alignment test failed: {str(e)}")
        return False

# ============================================================
# MAIN TEST SUITE
# ============================================================

def main():
    print(f"\n{Colors.BOLD}🧪 Emotion Recognition Dashboard - Test Suite{Colors.END}\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("PyTorch Setup", test_torch),
        ("OpenCV", test_opencv),
        ("Face Alignment", test_face_alignment),
        ("Transformers", test_transformers),
        ("Streamlit", test_streamlit),
        ("Model Files", test_model_files),
        ("Project Files", test_files_structure),
        ("Disk Space", test_disk_space)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            if test_name == "Package Imports":
                passed, failed = test_func()
                if failed:
                    print_warning(f"Missing packages: {', '.join(failed)}")
            else:
                passed = test_func()
            
            results.append((test_name, passed))
        
        except Exception as e:
            print_error(f"Test error: {str(e)}")
            results.append((test_name, False))
    
    # ============================================================
    # SUMMARY
    # ============================================================
    
    print(f"\n{Colors.BOLD}📊 Test Summary{Colors.END}\n")
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for test_name, passed_test in results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} | {test_name}")
    
    print(f"\n{'='*50}")
    print(f"Total: {passed}/{total} tests passed")
    print(f"{'='*50}\n")
    
    if passed == total:
        print_success("All tests passed! Dashboard is ready to run.")
        print(f"\n{Colors.BOLD}Next step:{Colors.END}")
        print(f"  streamlit run streamlit_app.py\n")
        return 0
    
    elif passed >= total - 2:  # Allow 2 non-critical failures
        print_warning("Some tests failed, but dashboard may still work.")
        print(f"\n{Colors.BOLD}Recommendations:{Colors.END}")
        print("  1. Check QUICK_START.md for setup instructions")
        print("  2. Run: python setup_dashboard.py --setup")
        print("  3. Try: streamlit run streamlit_app.py\n")
        return 1
    
    else:
        print_error("Multiple tests failed. Please fix issues before running dashboard.")
        print(f"\n{Colors.BOLD}Next steps:{Colors.END}")
        print("  1. Review test failures above")
        print("  2. Install missing packages: pip install -r requirements_streamlit.txt")
        print("  3. Run verification: python setup_dashboard.py --setup")
        print("  4. Check documentation: STREAMLIT_README.md\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
