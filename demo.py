"""
Script de demostración del sistema refactorizado.
Muestra cómo usar las nuevas clases modulares.
"""

def demo_models():
    """Demuestra el uso de los modelos."""
    print("🤖 DEMOSTRACIÓN DE MODELOS")
    print("="*40)
    
    from models import SAM2Model, MedSAM2Model, MobileSAMModel
    
    # Crear modelos (sin cargar weights para la demo)
    sam2 = SAM2Model(variant="tiny")
    print(f"✅ SAM2 creado: {sam2.model_name}")
    print(f"   Variantes disponibles: {list(SAM2Model.AVAILABLE_VARIANTS.keys())}")
    
    medsam2 = MedSAM2Model(variant="default")
    print(f"✅ MedSAM2 creado: {medsam2.model_name}")
    print(f"   Variantes disponibles: {list(MedSAM2Model.AVAILABLE_VARIANTS.keys())}")
    
    mobilesam = MobileSAMModel(variant="default")
    print(f"✅ MobileSAM creado: {mobilesam.model_name}")
    print(f"   Variantes disponibles: {list(MobileSAMModel.AVAILABLE_VARIANTS.keys())}")

def demo_trainers():
    """Demuestra el uso de los entrenadores."""
    print("\n⚙️  DEMOSTRACIÓN DE ENTRENADORES")
    print("="*40)
    
    from models import SAM2Model
    from trainers import BaselineTrainer, LoRATrainer, QLoRATrainer
    
    # Crear un modelo dummy para la demo
    model = SAM2Model(variant="tiny")
    
    # Mostrar entrenadores disponibles
    trainers = [
        ("Baseline", BaselineTrainer),
        ("LoRA", LoRATrainer), 
        ("QLoRA", QLoRATrainer)
    ]
    
    for name, trainer_class in trainers:
        trainer = trainer_class(model, learning_rate=1e-4)
        print(f"✅ {name}Trainer creado")
        print(f"   Clase: {trainer_class.__name__}")
        print(f"   Learning rate: {trainer.learning_rate}")

def demo_datasets():
    """Demuestra el uso de los datasets."""
    print("\n📊 DEMOSTRACIÓN DE DATASETS") 
    print("="*40)
    
    from datasets import list_available_datasets, create_dataset
    
    # Listar datasets disponibles
    datasets = list_available_datasets()
    print(f"✅ Datasets disponibles: {datasets}")
    
    # Mostrar cómo crear datasets (sin path real para la demo)
    print("\n📁 Ejemplo de creación de datasets:")
    print("   cataract_dataset = create_dataset('cataract', '/path/to/data')")
    print("   retinopathy_dataset = create_dataset('retinopathy', '/path/to/data')")

def demo_training_workflow():
    """Demuestra el flujo completo de entrenamiento."""
    print("\n🚀 FLUJO DE ENTRENAMIENTO COMPLETO")
    print("="*40)
    
    workflow = """
1. Crear modelo:
   model = SAM2Model(variant="tiny")
   model.load_model()
   model.load_processor()

2. Crear dataset:
   dataset = create_dataset("cataract", "/path/to/data")
   dataloader = DataLoader(dataset, batch_size=4)

3. Crear entrenador:
   trainer = LoRATrainer(model, learning_rate=1e-4)
   trainer.setup_model_for_training()
   trainer.setup_optimizer()

4. Entrenar:
   metrics = trainer.train(dataloader, epochs=5)

5. Guardar:
   trainer.save_checkpoint("./output")
"""
    print(workflow)

def demo_command_examples():
    """Muestra ejemplos de comandos."""
    print("\n💻 EJEMPLOS DE COMANDOS")
    print("="*40)
    
    examples = [
        ("Entrenamiento básico", 
         "python train.py --model sam2 --method lora --dataset cataract --dataset-root ./data/cataract"),
        
        ("Entrenamiento con configuración personalizada",
         "python train.py --model medsam2 --method qlora --dataset retinopathy --dataset-root ./data/retinopathy --epochs 10 --batch-size 4 --lora-r 32"),
        
        ("Benchmark completo",
         "python benchmark.py --dataset-root ./data --epochs 3"),
        
        ("Benchmark específico",
         "python benchmark.py --dataset-root ./data --models sam2 medsam2 --methods lora qlora")
    ]
    
    for name, command in examples:
        print(f"\n📝 {name}:")
        print(f"   {command}")

def main():
    """Ejecuta la demostración completa."""
    print("🎉 DEMOSTRACIÓN DEL SISTEMA REFACTORIZADO")
    print("="*60)
    print("Sistema modular para fine-tuning de modelos SAM")
    print("Usando Hugging Face Transformers y arquitectura POO")
    print("="*60)
    
    try:
        demo_models()
        demo_trainers()
        demo_datasets()
        demo_training_workflow()
        demo_command_examples()
        
        print("\n✨ CARACTERÍSTICAS PRINCIPALES")
        print("="*40)
        features = [
            "✅ Modular y orientado a objetos",
            "✅ Usa Hugging Face Transformers oficiales",
            "✅ Soporte para Baseline, LoRA y QLoRA",
            "✅ Datasets flexibles (COCO y estándar)",
            "✅ Benchmark automatizado",
            "✅ Manejo robusto de errores",
            "✅ Fácil extensibilidad"
        ]
        
        for feature in features:
            print(f"   {feature}")
            
        print("\n🎯 PRÓXIMOS PASOS")
        print("="*40)
        print("1. Configurar tus datasets en ./data/")
        print("2. Ejecutar: python train.py --list-models")
        print("3. Ejecutar: python train.py --list-datasets")
        print("4. Entrenar tu primer modelo:")
        print("   python train.py --model sam2 --method lora --dataset cataract --dataset-root ./data/cataract")
        
    except Exception as e:
        print(f"❌ Error en la demostración: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
