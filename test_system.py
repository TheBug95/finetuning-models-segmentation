"""
Script de prueba para verificar que el sistema funciona correctamente.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Prueba que todos los imports funcionen."""
    print("🔍 Probando imports...")
    
    try:
        # Test models
        from models import SAM2Model, MedSAM2Model, MobileSAMModel
        print("✅ Models imported successfully")
        
        # Test trainers
        from trainers import BaselineTrainer, LoRATrainer, QLoRATrainer
        print("✅ Trainers imported successfully")
        
        # Test datasets
        from datasets import create_dataset, list_available_datasets
        print("✅ Datasets imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Prueba la creación de modelos sin cargar weights."""
    print("\n🔍 Probando creación de modelos...")
    
    try:
        # Test SAM2Model
        from models import SAM2Model
        sam2 = SAM2Model(variant="tiny")
        print(f"✅ SAM2Model creado: {sam2.model_name}")
        
        # Test MedSAM2Model
        from models import MedSAM2Model
        medsam2 = MedSAM2Model(variant="default")
        print(f"✅ MedSAM2Model creado: {medsam2.model_name}")
        
        # Test MobileSAMModel
        from models import MobileSAMModel
        mobilesam = MobileSAMModel(variant="default")
        print(f"✅ MobileSAMModel creado: {mobilesam.model_name}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        traceback.print_exc()
        return False

def test_dataset_factory():
    """Prueba el factory de datasets."""
    print("\n🔍 Probando dataset factory...")
    
    try:
        from datasets import list_available_datasets
        datasets = list_available_datasets()
        print(f"✅ Datasets disponibles: {datasets}")
        
        return True
    except Exception as e:
        print(f"❌ Dataset factory error: {e}")
        traceback.print_exc()
        return False

def test_dependencies():
    """Prueba dependencias críticas."""
    print("\n🔍 Probando dependencias críticas...")
    
    dependencies = [
        "torch",
        "transformers", 
        "peft",
        "PIL",
        "numpy",
        "pandas"
    ]
    
    missing = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - MISSING")
            missing.append(dep)
    
    if missing:
        print(f"\n⚠️  Dependencias faltantes: {missing}")
        print("Ejecuta: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Ejecuta todas las pruebas."""
    print("🧪 EJECUTANDO PRUEBAS DEL SISTEMA")
    print("="*50)
    
    tests = [
        ("Dependencias", test_dependencies),
        ("Imports", test_imports),
        ("Creación de Modelos", test_model_creation),
        ("Dataset Factory", test_dataset_factory)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Error en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen
    print(f"\n📊 RESUMEN DE PRUEBAS")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Resultado: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! El sistema está listo.")
        return 0
    else:
        print("⚠️  Algunas pruebas fallaron. Revisa los errores arriba.")
        return 1

if __name__ == "__main__":
    exit(main())
