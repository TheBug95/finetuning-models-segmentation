import argparse
import os
import torch
from torch.utils.data import DataLoader
from datasets import get_dataset
from transformers import SamModel, SamProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Updated model IDs that actually exist in Hugging Face
MODEL_IDS = {
    "sam2": "facebook/sam2-hiera-tiny",  # Using available SAM2 model
    "medsam": "flaviagiammarino/medsam-vit-base",  # Available MedSAM model
    "mobilesam": "dhkim2810/MobileSAM",  # Available MobileSAM model
}

def create_model(model_key: str, method: str) -> SamModel:
    """Create and configure the model based on the specified method.
    
    Parameters
    ----------
    model_key : str
        Key for the model type
    method : str
        Training method: 'baseline', 'lora', or 'qlora'
        
    Returns
    -------
    SamModel
        Configured model ready for training
    """
    if model_key not in MODEL_IDS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_IDS.keys())}")
    
    model_id = MODEL_IDS[model_key]
    print(f"Loading model: {model_id}")
    
    try:
        if method == "qlora":
            model = SamModel.from_pretrained(
                model_id,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.float16,
            )
            model = prepare_model_for_kbit_training(model)
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="VISION",
            )
            model = get_peft_model(model, peft_config)
            print("Model configured with QLoRA")
            
        elif method == "lora":
            model = SamModel.from_pretrained(model_id)
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="VISION",
            )
            model = get_peft_model(model, peft_config)
            print("Model configured with LoRA")
            
        else:  # baseline
            model = SamModel.from_pretrained(model_id)
            print("Model loaded in baseline mode")
            
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_id}: {e}")
    
    return model

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

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SAM-based models on medical datasets.")
    parser.add_argument("--model", choices=list(MODEL_IDS.keys()), required=True,
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
    
    args = parser.parse_args()
    
    try:
        validate_args(args)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    try:
        model = create_model(args.model, args.method)
        model.to(device)
    except Exception as e:
        print(f"Error creating model: {e}")
        return 1
    
    # Create processor
    try:
        processor = SamProcessor.from_pretrained(MODEL_IDS[args.model])
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
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    model.train()
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        num_batches = 0
        
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
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:  # Log every 10 batches
                    print(f"Epoch {epoch + 1}/{args.epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
    
    # Save model
    if args.output_dir:
        save_dir = args.output_dir
    else:
        save_dir = f"finetuned-{args.model}-{args.method}-{args.dataset}"
    
    try:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        print(f"Model and processor saved to: {save_dir}")
    except Exception as e:
        print(f"Error saving model: {e}")
        return 1
    
    print("Training completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
