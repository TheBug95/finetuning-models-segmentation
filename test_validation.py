#!/usr/bin/env python3
"""
Script de validación ligero para verificar la estructura del proyecto.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that core imports work correctly."""
    print("Testing basic imports...")
    try:
        from datasets import get_dataset
        from datasets.common import build_medical_dataset
        from datasets.cataract import build_cataract_dataset
        from datasets.retinopathy import build_retinopathy_dataset
        print("✅ Dataset imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_manager():
    """Test model manager functionality without heavy dependencies."""
    print("\nTesting model manager...")
    try:
        from models import SAM2Model, MedSAM2Model, MobileSAMModel
        
        # Test basic model creation
        sam2 = SAM2Model(variant="tiny")
        print(f"✅ SAM2Model created: {sam2.model_name}")
        
        medsam2 = MedSAM2Model(variant="default") 
        print(f"✅ MedSAM2Model created: {medsam2.model_name}")
        
        mobilesam = MobileSAMModel(variant="default")
        print(f"✅ MobileSAMModel created: {mobilesam.model_name}")
        
        print("✅ Model creation working correctly")
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation functions."""
    print("\nTesting dataset creation...")
    try:
        from datasets import get_dataset
        
        # Test dataset factory function error handling
        try:
            get_dataset("nonexistent", "/fake/path")
        except ValueError as e:
            print(f"✅ Dataset factory error handling works: {e}")
        
        # Test build_medical_dataset error handling
        from datasets.common import build_medical_dataset
        try:
            build_medical_dataset("/nonexistent/path", "test")
        except ValueError as e:
            print(f"✅ Medical dataset error handling works: {e}")
        
        print("✅ Dataset creation functions working correctly")
        return True
    except Exception as e:
        print(f"❌ Dataset creation error: {e}")
        return False

def test_project_structure():
    """Test that all required files exist."""
    print("\nTesting project structure...")
    required_files = [
        "train.py",
        "benchmark.py", 
        "requirements.txt",
        "README.md",
        "datasets/__init__.py",
        "datasets/common.py",
        "datasets/cataract.py",
        "datasets/retinopathy.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def test_code_syntax():
    """Test that all Python files have valid syntax."""
    print("\nTesting code syntax...")
    python_files = [
        "train.py",
        "benchmark.py",
        "datasets/__init__.py", 
        "datasets/common.py",
        "datasets/cataract.py",
        "datasets/retinopathy.py",
    ]
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, file_path, 'exec')
        except Exception as e:
            print(f"❌ Syntax error in {file_path}: {e}")
            return False
    
    print("✅ All Python files have valid syntax")
    return True

def main():
    """Run all validation tests."""
    print("🧪 Running lightweight project validation tests...\n")
    
    tests = [
        test_project_structure,
        test_code_syntax,
        test_imports,
        test_model_manager,
        test_dataset_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project structure is correct.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
