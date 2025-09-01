"""
Script principal para fine-tuning de modelos SAM.
Versión refactorizada usando arquitectura modular y Hugging Face Transformers.
"""

import argparse
import os
import time
import json

import torch
from torch.utils.data import DataLoader

# Imports locales
from models import SAM2Model, MedSAM2Model, MobileSAMModel
from trainers import BaselineTrainer, LoRATrainer, QLoRATrainer
from datasets import create_dataset


def get_model_class(model_name: str):
    """Retorna la clase del modelo correspondiente."""
    model_classes = {
        "sam2": SAM2Model,
        "medsam2": MedSAM2Model,
        "mobilesam": MobileSAMModel
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Modelo no soportado: {model_name}. "
                        f"Disponibles: {list(model_classes.keys())}")
    
    return model_classes[model_name]


def get_trainer_class(method: str):
    """Retorna la clase del entrenador correspondiente."""
    trainer_classes = {
        "baseline": BaselineTrainer,
        "lora": LoRATrainer,
        "qlora": QLoRATrainer
    }
    
    if method not in trainer_classes:
        raise ValueError(f"Método no soportado: {method}. "
                        f"Disponibles: {list(trainer_classes.keys())}")
    
    return trainer_classes[method]


def setup_model(model_name: str, variant: str = None, cache_dir: str = None):
    """
    Configura y carga el modelo.
    
    Args:
        model_name: Nombre del modelo
        variant: Variante del modelo
        cache_dir: Directorio de cache
        
    Returns:
        Instancia del modelo configurada
    """
    print(f"🔧 Configurando modelo: {model_name}")
    
    model_class = get_model_class(model_name)
    
    # Crear instancia del modelo
    if variant:
        model = model_class(variant=variant, cache_dir=cache_dir)
    else:
        model = model_class(cache_dir=cache_dir)
    
    # Cargar modelo y procesador
    model.load_model()
    model.load_processor()
    model.to_device()
    
    print(f"✅ Modelo {model_name} configurado exitosamente")
    return model


def setup_trainer(model, method: str, learning_rate: float, **trainer_kwargs):
    """
    Configura el entrenador.
    
    Args:
        model: Modelo a entrenar
        method: Método de entrenamiento
        learning_rate: Tasa de aprendizaje
        **trainer_kwargs: Argumentos adicionales para el entrenador
        
    Returns:
        Instancia del entrenador configurada
    """
    print(f"🔧 Configurando entrenador: {method}")
    
    trainer_class = get_trainer_class(method)
    trainer = trainer_class(model, learning_rate=learning_rate, **trainer_kwargs)
    
    # Configurar modelo para entrenamiento
    trainer.setup_model_for_training()
    trainer.setup_optimizer()
    
    print(f"✅ Entrenador {method} configurado exitosamente")
    return trainer


def setup_dataset(dataset_name: str, dataset_root: str, batch_size: int, **dataset_kwargs):
    """
    Configura el dataset y dataloader.
    
    Args:
        dataset_name: Nombre del dataset
        dataset_root: Directorio raíz del dataset
        batch_size: Tamaño del batch
        **dataset_kwargs: Argumentos adicionales para el dataset
        
    Returns:
        DataLoader configurado
    """
    print(f"🔧 Configurando dataset: {dataset_name}")
    
    # Crear dataset
    dataset = create_dataset(dataset_name, dataset_root, **dataset_kwargs)
    
    # Crear dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"✅ Dataset configurado: {len(dataset)} muestras, batch_size={batch_size}")
    return dataloader


def save_training_results(trainer, model, output_dir: str, args):
    """Guarda los resultados del entrenamiento."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar modelo
    model.save_model(output_dir)
    
    # Guardar resumen de entrenamiento
    training_summary = trainer.get_training_summary()
    training_summary["command_line_args"] = vars(args)
    
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(training_summary, f, indent=2)
    
    # Guardar información del modelo
    model_info = model.info
    with open(os.path.join(output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"💾 Resultados guardados en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning modular de modelos SAM para segmentación médica",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Argumentos del modelo
    parser.add_argument("--model", 
                       choices=["sam2", "medsam2", "mobilesam"],
                       required=True,
                       help="Modelo a entrenar")
    parser.add_argument("--variant",
                       help="Variante del modelo (opcional)")
    
    # Argumentos del método de entrenamiento
    parser.add_argument("--method",
                       choices=["baseline", "lora", "qlora"],
                       default="lora",
                       help="Método de fine-tuning")
    
    # Argumentos del dataset
    parser.add_argument("--dataset",
                       choices=["cataract", "retinopathy"],
                       required=True,
                       help="Dataset a usar")
    parser.add_argument("--dataset-root",
                       required=True,
                       help="Directorio raíz del dataset")
    parser.add_argument("--split",
                       default="train",
                       help="Split del dataset")
    
    # Argumentos de entrenamiento
    parser.add_argument("--epochs",
                       type=int,
                       default=5,
                       help="Número de épocas")
    parser.add_argument("--batch-size",
                       type=int,
                       default=2,
                       help="Tamaño del batch")
    parser.add_argument("--learning-rate",
                       type=float,
                       default=1e-4,
                       help="Tasa de aprendizaje")
    parser.add_argument("--weight-decay",
                       type=float,
                       default=0.01,
                       help="Weight decay")
    
    # Argumentos de LoRA/QLoRA
    parser.add_argument("--lora-r",
                       type=int,
                       default=16,
                       help="Rango de LoRA")
    parser.add_argument("--lora-alpha",
                       type=int,
                       default=32,
                       help="Alpha de LoRA")
    parser.add_argument("--lora-dropout",
                       type=float,
                       default=0.1,
                       help="Dropout de LoRA")
    
    # Argumentos de salida
    parser.add_argument("--output-dir",
                       help="Directorio de salida (se genera automáticamente si no se especifica)")
    parser.add_argument("--cache-dir",
                       default="./cache",
                       help="Directorio de cache para modelos")
    
    # Argumentos de utilidad
    parser.add_argument("--list-models",
                       action="store_true",
                       help="Listar modelos disponibles")
    parser.add_argument("--list-datasets",
                       action="store_true",
                       help="Listar datasets disponibles")
    
    args = parser.parse_args()
    
    # Comandos de utilidad
    if args.list_models:
        print("Modelos disponibles:")
        print("  sam2: SAM2 con variantes tiny, base, large, huge")
        print("  medsam2: MedSAM2 con variantes default, vit_base, medsam_mix")
        print("  mobilesam: MobileSAM con variantes default, qualcomm, dhkim, v2")
        return 0
    
    if args.list_datasets:
        from datasets import list_available_datasets
        print("Datasets disponibles:")
        for dataset in list_available_datasets():
            print(f"  {dataset}")
        return 0
    
    # Validaciones
    if not os.path.exists(args.dataset_root):
        print(f"❌ Error: Dataset root no existe: {args.dataset_root}")
        return 1
    
    # Configurar output directory
    if not args.output_dir:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/{args.model}_{args.method}_{args.dataset}_{timestamp}"
    
    print("🚀 INICIANDO FINE-TUNING MODULAR")
    print("="*60)
    print(f"🤖 Modelo: {args.model} ({args.variant or 'default'})")
    print(f"⚙️  Método: {args.method}")
    print(f"📊 Dataset: {args.dataset}")
    print(f"📁 Dataset root: {args.dataset_root}")
    print(f"📈 Épocas: {args.epochs}")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"📈 Learning rate: {args.learning_rate}")
    print("="*60)
    
    try:
        # 1. Configurar modelo
        model = setup_model(args.model, args.variant, args.cache_dir)
        
        # 2. Configurar dataset
        dataloader = setup_dataset(
            args.dataset,
            args.dataset_root,
            args.batch_size,
            split=args.split
        )
        
        # 3. Configurar entrenador
        trainer_kwargs = {}
        if args.method in ["lora", "qlora"]:
            trainer_kwargs.update({
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout
            })
        
        trainer = setup_trainer(
            model,
            args.method,
            args.learning_rate,
            weight_decay=args.weight_decay,
            **trainer_kwargs
        )
        
        # 4. Entrenar
        print("\n🎯 INICIANDO ENTRENAMIENTO")
        print("="*60)
        
        start_time = time.time()
        metrics = trainer.train(dataloader, args.epochs, args.output_dir)
        end_time = time.time()
        
        # 5. Guardar resultados
        save_training_results(trainer, model, args.output_dir, args)
        
        # 6. Resumen final
        print("\n🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*60)
        print(f"⏱️  Tiempo total: {end_time - start_time:.2f}s")
        print(f"📈 Mejor loss: {metrics.best_loss:.6f}")
        print(f"📁 Resultados en: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print("\n❌ ERROR EN EL ENTRENAMIENTO")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
