#!/usr/bin/env python3
"""
Helper script for setting up and testing the Streamlit Emotion Recognition Dashboard
Handles model verification, dependencies, and initial configuration
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Tuple

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

# ============================================================
# SYSTEM CHECKS
# ============================================================

def check_python_version() -> bool:
    """Check if Python version meets requirements"""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version >= required_version:
        print_success(f"Python {current_version[0]}.{current_version[1]} ({'OK' if current_version >= required_version else 'Too old'})")
        return True
    else:
        print_error(f"Python {required_version[0]}.{required_version[1]}+ required, got {current_version[0]}.{current_version[1]}")
        return False

def check_gpu_availability() -> bool:
    """Check if GPU is available"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        print_success(f"GPU Detected: {device_name} (Device count: {device_count})")
        
        # Check VRAM
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_info(f"GPU VRAM: {vram_gb:.2f}GB")
        
        if vram_gb < 4:
            print_warning(f"GPU VRAM is low ({vram_gb:.2f}GB). Recommended: 4GB+")
        
        return True
    else:
        print_warning("No GPU detected. Will use CPU (slower inference)")
        return False

def check_disk_space(required_gb: float = 6.0) -> bool:
    """Check available disk space"""
    import shutil
    
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    
    if free_gb >= required_gb:
        print_success(f"Disk Space: {free_gb:.2f}GB available")
        return True
    else:
        print_error(f"Insufficient disk space. Required: {required_gb}GB, Available: {free_gb:.2f}GB")
        return False

# ============================================================
# MODEL VERIFICATION
# ============================================================

def check_package(package_name: str) -> bool:
    """Check if a Python package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def check_dependencies() -> Tuple[bool, list]:
    """Check all required dependencies"""
    required_packages = {
        'streamlit': 'Streamlit',
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'pandas': 'Pandas'
    }
    
    missing = []
    
    for module, display_name in required_packages.items():
        if check_package(module):
            print_success(f"{display_name}")
        else:
            print_warning(f"{display_name} (not installed)")
            missing.append(module)
    
    return len(missing) == 0, missing

def check_base_model() -> bool:
    """Check if BLIP-2 base model can be loaded"""
    print_info("Testing base model load (this may take a moment)...")
    
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        print("  Loading processor...", end=" ")
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        print_success("Done")
        
        print("  Loading model...", end=" ")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            device_map="auto"
        )
        print_success("Done")
        
        return True
    
    except Exception as e:
        print_error(f"Failed to load base model: {str(e)}")
        return False

def check_lora_model(model_path: str) -> bool:
    """Check if LoRA adapters exist and can be loaded"""
    print_info(f"Checking LoRA adapters at: {model_path}")
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        print_error(f"Model path does not exist: {model_path}")
        return False
    
    required_files = [
        'adapter_config.json',
        'adapter_model.bin'
    ]
    
    all_present = True
    for file in required_files:
        file_path = model_path / file
        if file_path.exists():
            print_success(f"Found: {file}")
        else:
            print_error(f"Missing: {file}")
            all_present = False
    
    if all_present:
        print_info("Attempting to load LoRA adapters...")
        try:
            from peft import PeftModel
            from transformers import Blip2ForConditionalGeneration
            
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b"
            )
            model = PeftModel.from_pretrained(model, model_path)
            print_success("LoRA adapters loaded successfully")
            return True
        
        except Exception as e:
            print_error(f"Failed to load LoRA adapters: {str(e)}")
            return False
    
    return False

# ============================================================
# FILE STRUCTURE
# ============================================================

def check_file_structure() -> bool:
    """Check if project files are in correct structure"""
    print_info("Checking project file structure...")
    
    required_files = [
        'streamlit_app.py',
        'requirements_streamlit.txt',
        'STREAMLIT_README.md'
    ]
    
    all_present = True
    for file in required_files:
        if os.path.exists(file):
            print_success(f"Found: {file}")
        else:
            print_warning(f"Missing: {file}")
            all_present = False
    
    return all_present

# ============================================================
# SETUP FUNCTIONS
# ============================================================

def install_dependencies() -> bool:
    """Install missing dependencies"""
    print_info("Installing dependencies...")
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', 'requirements_streamlit.txt'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print_success("Dependencies installed successfully")
            return True
        else:
            print_error(f"Dependency installation failed: {result.stderr}")
            return False
    
    except Exception as e:
        print_error(f"Error during installation: {str(e)}")
        return False

def test_inference(model_path: str) -> bool:
    """Test model inference with a dummy image"""
    print_info("Testing model inference...")
    
    try:
        from PIL import Image
        import numpy as np
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        from peft import PeftModel
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create dummy image
        print("  Creating test image...", end=" ")
        dummy_image = Image.new('RGB', (336, 336), color='white')
        print_success("Done")
        
        # Load model
        print("  Loading model...", end=" ")
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        if os.path.exists(model_path):
            model = PeftModel.from_pretrained(model, model_path)
        
        print_success("Done")
        
        # Test inference
        print("  Running inference...", end=" ")
        prompt = "What is in this image?"
        inputs = processor(images=dummy_image, text=prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
        
        output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print_success("Done")
        
        print_info(f"Test output: {output[:100]}...")
        return True
    
    except Exception as e:
        print_error(f"Inference test failed: {str(e)}")
        return False

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Setup and verify Streamlit Emotion Recognition Dashboard'
    )
    parser.add_argument(
        '--model-path',
        default='./blip2-emotion-rafce-final',
        help='Path to fine-tuned LoRA model (default: ./blip2-emotion-rafce-final)'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check requirements, do not install'
    )
    parser.add_argument(
        '--test-inference',
        action='store_true',
        help='Test model inference'
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Complete setup (install dependencies)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}🎭 Streamlit Emotion Recognition Dashboard - Setup & Verification{Colors.END}\n")
    
    # ============================================================
    # SYSTEM CHECKS
    # ============================================================
    
    print(f"{Colors.BOLD}System Requirements:{Colors.END}")
    checks = [
        check_python_version(),
        check_gpu_availability(),
        check_disk_space()
    ]
    
    if not all(checks):
        print_warning("Some system requirements not met")
    
    # ============================================================
    # DEPENDENCY CHECKS
    # ============================================================
    
    print(f"\n{Colors.BOLD}Python Dependencies:{Colors.END}")
    deps_ok, missing = check_dependencies()
    
    if not deps_ok:
        if args.setup:
            print("\nInstalling missing dependencies...")
            if install_dependencies():
                print_success("Setup completed successfully")
            else:
                print_error("Setup failed")
                return False
        else:
            print_warning(f"Missing packages: {', '.join(missing)}")
            print_info("Run with --setup flag to install dependencies")
    
    # ============================================================
    # FILE STRUCTURE
    # ============================================================
    
    print(f"\n{Colors.BOLD}Project Files:{Colors.END}")
    check_file_structure()
    
    # ============================================================
    # MODEL VERIFICATION
    # ============================================================
    
    print(f"\n{Colors.BOLD}Model Verification:{Colors.END}")
    
    if not args.check_only:
        base_model_ok = check_base_model()
        lora_model_ok = check_lora_model(args.model_path)
        
        if base_model_ok and lora_model_ok:
            print_success("All model checks passed")
        else:
            print_warning("Some model checks failed")
    
    # ============================================================
    # INFERENCE TEST
    # ============================================================
    
    if args.test_inference:
        print(f"\n{Colors.BOLD}Inference Test:{Colors.END}")
        test_inference(args.model_path)
    
    # ============================================================
    # SUMMARY & NEXT STEPS
    # ============================================================
    
    print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
    print(f"  1. Ensure fine-tuned model is in: {args.model_path}/")
    print(f"  2. Run: {Colors.BOLD}streamlit run streamlit_app.py{Colors.END}")
    print(f"  3. Open: http://localhost:8501")
    
    print(f"\n{Colors.BOLD}📖 For more help, see: STREAMLIT_README.md{Colors.END}\n")

if __name__ == "__main__":
    main()
