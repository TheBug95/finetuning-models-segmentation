import argparse
import torch
from torch.utils.data import DataLoader
from datasets import get_dataset
from transformers import SamModel, SamProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_IDS = {
    "sam2": "facebook/sam2-hiera-small",
    "medsam2": "facebook/medsam2-hiera-small",
    "mobilesam": "facebook/mobile-sam",
}

def create_model(model_key: str, method: str) -> SamModel:
    model_id = MODEL_IDS[model_key]
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
    else:
        model = SamModel.from_pretrained(model_id)
    return model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SAM-based models on medical datasets.")
    parser.add_argument("--model", choices=list(MODEL_IDS.keys()), required=True)
    parser.add_argument("--method", choices=["baseline", "lora", "qlora"], default="baseline")
    parser.add_argument("--dataset", choices=["cataract", "retinopathy"], required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model, args.method)
    model.to(device)

    processor = SamProcessor.from_pretrained(MODEL_IDS[args.model])
    dataset = get_dataset(args.dataset, args.dataset_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        for images, masks in dataloader:
            inputs = processor(images=list(images), segmentation_maps=list(masks), return_tensors="pt").to(device)
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch + 1}: loss={loss.item():.4f}")

    save_dir = f"finetuned-{args.model}-{args.method}-{args.dataset}"
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

if __name__ == "__main__":
    main()
