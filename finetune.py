import argparse
import os
import torch
from torch.utils.data import DataLoader
from datasets import get_dataset
from transformers import SamModel, SamProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from model_manager import SAMModelManager

# Configuración específica para SAM2, MedSAM2 y MobileSAM2
MODEL_CONFIG = {
    "sam2": {
        "hf_id": "facebook/sam2-hiera-tiny",     # SAM2 oficial de Facebook
        "manager_family": "sam",                  # Usar SAM base en manager
        "manager_variant": "vit_b",              # Variante base
        "description": "SAM2 - Next generation Segment Anything Model"
    },
    "medsam2": {
        "hf_id": "facebook/sam2-hiera-small",    # Usar SAM2 como base para MedSAM2
        "manager_family": "medsam",              # MedSAM en manager
        "manager_variant": "medsam2_latest",     # MedSAM2 latest
        "description": "MedSAM2 - Medical specialized SAM2"
    },
    "mobilesam2": {
        "hf_id": "facebook/sam2-hiera-tiny",     # Usar SAM2 tiny como base para Mobile
        "manager_family": "mobilesam",           # MobileSAM en manager
        "manager_variant": "vit_t",              # Variante tiny
        "description": "MobileSAM2 - Lightweight SAM2 for mobile devices"
    }
}

def setup_model_with_manager(model_key: str, models_dir: str = "models") -> tuple[str, str]:
    """Setup and download model using SAMModelManager.
    
    Parameters
    ----------
    model_key : str
        Key for the model type
    models_dir : str
        Directory to store downloaded models
        
    Returns
    -------
    tuple[str, str]
        Tuple of (huggingface_model_id, local_checkpoint_path)
    """
    if model_key not in MODEL_CONFIG:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_CONFIG.keys())}")
    
    config = MODEL_CONFIG[model_key]
    print(f"Setting up {config['description']}...")
    
    # Initialize SAMModelManager
    try:
        manager = SAMModelManager(models_dir)
        print(f"SAMModelManager initialized with directory: {models_dir}")
        
        # Download and setup the model
        checkpoint_path = manager.setup(
            family=config["manager_family"],
            variant=config["manager_variant"],
            install=True,
            force=False
        )
        
        print(f"✅ Model checkpoint ready at: {checkpoint_path}")
        return config["hf_id"], str(checkpoint_path)
        
    except Exception as e:
        print(f"⚠️  Warning: SAMModelManager setup failed: {e}")
        print(f"   Falling back to direct Hugging Face download for {config['hf_id']}")
        return config["hf_id"], None

def create_model(model_key: str, method: str, models_dir: str = "models") -> tuple[SamModel, str]:
    """Create and configure the model based on the specified method.
    
    Parameters
    ----------
    model_key : str
        Key for the model type (sam2, medsam2, mobilesam2)
    method : str
        Training method: 'baseline', 'lora', or 'qlora'
    models_dir : str
        Directory to store downloaded models
        
    Returns
    -------
    tuple[SamModel, str]
        Tuple of (configured_model, huggingface_model_id)
    """
    if model_key not in MODEL_CONFIG:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_CONFIG.keys())}")
    
    # Setup model using SAMModelManager
    hf_model_id, checkpoint_path = setup_model_with_manager(model_key, models_dir)
    print(f"Loading model from Hugging Face: {hf_model_id}")
    
    try:
        if method == "baseline":
            # Fine-tuning completo sin técnicas de eficiencia
            print("🔧 Configurando modelo para fine-tuning BASELINE (completo)")
            model = SamModel.from_pretrained(
                hf_model_id,
                torch_dtype=torch.float32,  # Usar float32 para baseline
            )
            
            # Descongelar todas las capas para fine-tuning completo
            for param in model.parameters():
                param.requires_grad = True
            
            print("   ✅ Todas las capas descongeladas para entrenamiento completo")
            print("   📊 Parámetros entrenables:", sum(p.numel() for p in model.parameters() if p.requires_grad))
            
        elif method == "qlora":
            print("🔧 Configurando modelo para fine-tuning con QLoRA")
            model = SamModel.from_pretrained(
                hf_model_id,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.float16,
            )
            model = prepare_model_for_kbit_training(model)
            
            # Configuración específica para SAM2/MedSAM2/MobileSAM2
            target_modules = [
                "q_proj", "k_proj", "v_proj", "out_proj",  # Attention modules
                "linear1", "linear2",                       # FFN modules
            ]
            
            peft_config = LoraConfig(
                r=16,                    # Rango más alto para mejor calidad
                lora_alpha=32,           # Alpha proporcional
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type="VISION",
            )
            model = get_peft_model(model, peft_config)
            print("   ✅ QLoRA configurado con cuantización 4-bit")
            
        elif method == "lora":
            print("🔧 Configurando modelo para fine-tuning con LoRA")
            model = SamModel.from_pretrained(
                hf_model_id,
                torch_dtype=torch.float16,  # float16 para eficiencia
            )
            
            # Configuración específica para SAM2/MedSAM2/MobileSAM2
            target_modules = [
                "q_proj", "k_proj", "v_proj", "out_proj",  # Attention modules
                "linear1", "linear2",                       # FFN modules
            ]
            
            peft_config = LoraConfig(
                r=16,                    # Rango más alto para mejor calidad
                lora_alpha=32,           # Alpha proporcional
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type="VISION",
            )
            model = get_peft_model(model, peft_config)
            print("   ✅ LoRA configurado sin cuantización")
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Mostrar estadísticas del modelo
        if method != "baseline":
            # Para LoRA/QLoRA mostrar parámetros entrenables vs total
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   📊 Parámetros entrenables: {trainable_params:,}")
            print(f"   📊 Parámetros totales: {total_params:,}")
            print(f"   📊 Porcentaje entrenable: {100 * trainable_params / total_params:.2f}%")
        
        # If we have a local checkpoint, we could optionally load weights here
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"✅ Native checkpoint available at: {checkpoint_path}")
            print("   (Using Hugging Face model for compatibility with transformers)")
            
    except Exception as e:
        raise RuntimeError(f"Error loading model {hf_model_id}: {e}")
    
    return model, hf_model_id

def validate_args(args):
    """Validate command line arguments."""
    if not os.path.exists(args.dataset_root):
        raise ValueError(f"Dataset root does not exist: {args.dataset_root}")
    
    if args.epochs <= 0:
        raise ValueError("Epochs must be positive")
    
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    if args.lr <= 0:
        raise ValueError("Learning rate must be positive")

def show_model_manager_status(models_dir: str = "models"):
    """Show SAMModelManager status and available models."""
    try:
        manager = SAMModelManager(models_dir)
        print("\n🔧 SAMModelManager Status:")
        print(f"   Models directory: {models_dir}")
        manager.list_supported()
        
        # Check what's already downloaded
        models_path = os.path.join(models_dir)
        if os.path.exists(models_path):
            downloaded_files = [f for f in os.listdir(models_path) 
                              if f.endswith(('.pth', '.pt', '.bin'))]
            if downloaded_files:
                print(f"\n📁 Already downloaded checkpoints:")
                for file in downloaded_files:
                    print(f"   - {file}")
            else:
                print(f"\n📁 No checkpoints found in {models_path}")
        else:
            print(f"\n📁 Models directory will be created: {models_path}")
            
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize SAMModelManager: {e}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SAM-based models on medical datasets.")
    parser.add_argument("--model", choices=list(MODEL_CONFIG.keys()), required=True,
                       help="Model to fine-tune")
    parser.add_argument("--method", choices=["baseline", "lora", "qlora"], default="baseline",
                       help="Training method")
    parser.add_argument("--dataset", choices=["cataract", "retinopathy"], required=True,
                       help="Dataset to use")
    parser.add_argument("--dataset-root", required=True,
                       help="Root directory of the dataset")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for saved model (auto-generated if not specified)")
    parser.add_argument("--models-dir", default="models",
                       help="Directory to store downloaded models")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    parser.add_argument("--show-manager-status", action="store_true",
                       help="Show SAMModelManager status and exit")
    
    args = parser.parse_args()
    
    # Show manager status if requested
    if args.show_manager_status:
        show_model_manager_status(args.models_dir)
        return 0
    
    # List models if requested
    if args.list_models:
        print("Available models for fine-tuning:")
        for key, config in MODEL_CONFIG.items():
            print(f"  {key}: {config['description']}")
            print(f"    └── HF Model: {config['hf_id']}")
            print(f"    └── Manager: {config['manager_family']}/{config['manager_variant']}")
        return 0
    
    try:
        validate_args(args)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model using integrated SAMModelManager
    try:
        model, hf_model_id = create_model(args.model, args.method, args.models_dir)
        model.to(device)
    except Exception as e:
        print(f"Error creating model: {e}")
        return 1
    
    # Create processor
    try:
        processor = SamProcessor.from_pretrained(hf_model_id)
    except Exception as e:
        print(f"Error loading processor: {e}")
        return 1
    
    # Load dataset
    try:
        dataset = get_dataset(args.dataset, args.dataset_root)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        print(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    # Set up optimizer con configuración específica por método
    if args.method == "baseline":
        # Para baseline usar learning rate más bajo debido a fine-tuning completo
        lr = args.lr * 0.1  # 10x menor para baseline
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        print(f"🔧 Optimizer configurado para BASELINE - LR: {lr:.2e} (reducido para estabilidad)")
    else:
        # Para LoRA/QLoRA usar learning rate normal
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        print(f"🔧 Optimizer configurado para {args.method.upper()} - LR: {args.lr:.2e}")
    
    # Training metrics tracking
    import time
    training_start_time = time.time()
    epoch_losses = []
    
    # Training loop mejorado para comparaciones
    model.train()
    print(f"\n🚀 Iniciando entrenamiento: {args.model.upper()} con método {args.method.upper()}")
    print("="*80)
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        num_batches = 0
        
        print(f"\n📈 Época {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            try:
                # Process inputs
                inputs = processor(
                    images=list(images), 
                    segmentation_maps=list(masks), 
                    return_tensors="pt"
                ).to(device)
                
                # Forward pass
                outputs = model(**inputs)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping para estabilidad (especialmente importante para baseline)
                if args.method == "baseline":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Log más detallado cada 5 batches
                if batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx:3d}: Loss = {loss.item():.6f}")
                    
            except Exception as e:
                print(f"❌ Error in training batch {batch_idx}: {e}")
                continue
        
        # Estadísticas de época
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - epoch_start_time
        epoch_losses.append(avg_loss)
        
        print(f"\n✅ Época {epoch + 1} completada:")
        print(f"   💯 Loss promedio: {avg_loss:.6f}")
        print(f"   ⏱️  Tiempo: {epoch_time:.2f}s")
        print(f"   📊 Batches procesados: {num_batches}")
        
        # Mostrar tendencia de loss
        if len(epoch_losses) > 1:
            improvement = epoch_losses[-2] - epoch_losses[-1]
            if improvement > 0:
                print(f"   📈 Mejora: -{improvement:.6f} ({improvement/epoch_losses[-2]*100:.2f}%)")
            else:
                print(f"   📉 Cambio: +{abs(improvement):.6f}")
    
    # Resumen final del entrenamiento
    total_training_time = time.time() - training_start_time
    print(f"\n🎯 RESUMEN DEL ENTRENAMIENTO")
    print("="*80)
    print(f"🤖 Modelo: {args.model.upper()}")
    print(f"⚙️  Método: {args.method.upper()}")
    print(f"📊 Dataset: {args.dataset}")
    print(f"⏱️  Tiempo total: {total_training_time:.2f}s ({total_training_time/60:.1f}min)")
    print(f"📈 Loss inicial: {epoch_losses[0]:.6f}")
    print(f"📈 Loss final: {epoch_losses[-1]:.6f}")
    
    if len(epoch_losses) > 1:
        total_improvement = epoch_losses[0] - epoch_losses[-1]
        print(f"📈 Mejora total: {total_improvement:.6f} ({total_improvement/epoch_losses[0]*100:.2f}%)")
    
    # Save model con información del método
    if args.output_dir:
        save_dir = args.output_dir
    else:
        save_dir = f"finetuned-{args.model}-{args.method}-{args.dataset}-epochs{args.epochs}"
    
    try:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        
        # Guardar métricas de entrenamiento
        import json
        training_info = {
            "model": args.model,
            "method": args.method,
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "training_time_seconds": total_training_time,
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1],
            "total_improvement": epoch_losses[0] - epoch_losses[-1] if len(epoch_losses) > 1 else 0
        }
        
        with open(os.path.join(save_dir, "training_metrics.json"), "w") as f:
            json.dump(training_info, f, indent=2)
        
        print(f"\n💾 Modelo y métricas guardadas en: {save_dir}")
        print(f"   📁 Archivos generados:")
        print(f"     - Model files (adapter_model.bin, config.json, etc.)")
        print(f"     - training_metrics.json (métricas de entrenamiento)")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return 1
    
    print("Training completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
