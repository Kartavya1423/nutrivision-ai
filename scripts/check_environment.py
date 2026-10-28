#!/usr/bin/env python3
"""
NutriVision AI — Environment & Hardware Compatibility Checker
Validates Python version, PyTorch, CUDA, OpenCV, NumPy, and Pillow.
"""

import sys
import os
import platform

def format_status(passed: bool) -> str:
    """Return colored status string if supported, else plain text."""
    if passed:
        return "[PASS] OK"
    return "[MISSING] NOT FOUND"

def check_environment() -> bool:
    print("=" * 65)
    print("      NutriVision AI - Environment & Dependency Checker       ")
    print("=" * 65)
    print(f"Operating System : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"Machine / Arch   : {platform.machine()}")
    print(f"Working Directory: {os.getcwd()}")
    print("-" * 65)

    all_critical_passed = True
    results = []

    # 1. Python Version Check
    py_major = sys.version_info.major
    py_minor = sys.version_info.minor
    py_micro = sys.version_info.micro
    py_ver_str = f"{py_major}.{py_minor}.{py_micro}"
    py_passed = (py_major == 3 and py_minor >= 11)
    results.append({
        "component": "Python (>= 3.11)",
        "version": py_ver_str,
        "status": "PASS" if py_passed else "FAIL",
        "notes": "Target: 3.11+" if py_passed else "Python 3.11+ required"
    })
    if not py_passed:
        all_critical_passed = False

    # 2. NumPy Check
    try:
        import numpy as np
        np_ver = np.__version__
        results.append({
            "component": "NumPy",
            "version": np_ver,
            "status": "PASS",
            "notes": "Fast array computations"
        })
    except ImportError:
        results.append({
            "component": "NumPy",
            "version": "None",
            "status": "MISSING",
            "notes": "Run: pip install numpy"
        })
        all_critical_passed = False

    # 3. Pillow Check
    try:
        import PIL
        pil_ver = PIL.__version__
        results.append({
            "component": "Pillow (PIL)",
            "version": pil_ver,
            "status": "PASS",
            "notes": "Image loading & preprocessing"
        })
    except ImportError:
        results.append({
            "component": "Pillow (PIL)",
            "version": "None",
            "status": "MISSING",
            "notes": "Run: pip install pillow"
        })
        all_critical_passed = False

    # 4. PyTorch Check
    torch_available = False
    cuda_available = False
    cuda_device_name = "N/A"
    try:
        import torch
        torch_ver = torch.__version__
        torch_available = True
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            cuda_notes = f"{cuda_device_name} ({device_count} GPU{'s' if device_count > 1 else ''})"
        else:
            cuda_notes = "CPU Mode (No CUDA GPU detected)"
        
        results.append({
            "component": "PyTorch",
            "version": torch_ver,
            "status": "PASS",
            "notes": "Deep learning framework"
        })
        results.append({
            "component": "CUDA Acceleration",
            "version": f"CUDA {torch.version.cuda}" if torch.version.cuda else "None",
            "status": "ENABLED" if cuda_available else "DISABLED (CPU)",
            "notes": cuda_notes
        })
    except ImportError:
        results.append({
            "component": "PyTorch",
            "version": "Not installed",
            "status": "MISSING",
            "notes": "Required for ML modules. Run: pip install torch torchvision"
        })
        results.append({
            "component": "CUDA Acceleration",
            "version": "N/A",
            "status": "UNKNOWN",
            "notes": "PyTorch not installed"
        })

    # 5. OpenCV Check
    try:
        import cv2
        cv_ver = cv2.__version__
        results.append({
            "component": "OpenCV (cv2)",
            "version": cv_ver,
            "status": "PASS",
            "notes": "Computer vision & image filtering"
        })
    except ImportError:
        results.append({
            "component": "OpenCV (cv2)",
            "version": "Not installed",
            "status": "MISSING",
            "notes": "Run: pip install opencv-python"
        })

    # 6. PyYAML Check (for config.yaml)
    try:
        import yaml
        yaml_ver = getattr(yaml, "__version__", "Available")
        results.append({
            "component": "PyYAML",
            "version": yaml_ver,
            "status": "PASS",
            "notes": "Configuration parsing"
        })
    except ImportError:
        results.append({
            "component": "PyYAML",
            "version": "Not installed",
            "status": "OPTIONAL",
            "notes": "Run: pip install pyyaml"
        })

    # Print Formatted Table
    print(f"{'Component':<22} | {'Version':<16} | {'Status':<14} | {'Notes'}")
    print("-" * 75)
    for r in results:
        print(f"{r['component']:<22} | {r['version']:<16} | {r['status']:<14} | {r['notes']}")
    print("-" * 75)

    # Summary Assessment
    print("\n[Environment Assessment]")
    if all_critical_passed:
        print("  Base environment (Python, NumPy, Pillow) is OPERATIONAL.")
    else:
        print("  One or more base dependencies are missing.")

    if not torch_available:
        print("  [Notice] PyTorch is not yet installed. For Module 0, this is normal.")
        print("    When ready for model modules, install via:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 (for CUDA)")
        print("    or: pip install torch torchvision (for CPU)")

    if not any(r['component'] == 'OpenCV (cv2)' and r['status'] == 'PASS' for r in results):
        print("  [Notice] OpenCV is not yet installed. Install via: pip install opencv-python")

    print("\nTo install all required packages at once:")
    print("  pip install -r requirements.txt")
    print("=" * 65)

    return all_critical_passed

if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
