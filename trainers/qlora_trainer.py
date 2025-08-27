"""
Entrenador para fine-tuning con QLoRA.
"""

import torch
from torch.optim import AdamW
from .lora_trainer import LoRATrainer


class QLoRATrainer(LoRATrainer):
    """Entrenador para fine-tuning con QLoRA (LoRA cuantizado)."""
    
    def __init__(self, 
                 model, 
                 learning_rate: float = 2e-4,  # LR un poco más alto para QLoRA
                 weight_decay: float = 0.01,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1):
        """
        Inicializa el entrenador QLoRA.
        
        Args:
            model: Modelo de segmentación
            learning_rate: Tasa de aprendizaje (más alta para QLoRA)
            weight_decay: Decay de pesos
            lora_r: Rango de LoRA
            lora_alpha: Alpha de LoRA
            lora_dropout: Dropout de LoRA
        """
        super().__init__(model, learning_rate, weight_decay, lora_r, lora_alpha, lora_dropout)
        
    def setup_model_for_training(self, **kwargs) -> None:
        """Configura el modelo para fine-tuning con QLoRA."""
        lora_config = {
            "r": kwargs.get("r", self.lora_r),
            "lora_alpha": kwargs.get("lora_alpha", self.lora_alpha),
            "lora_dropout": kwargs.get("lora_dropout", self.lora_dropout),
            "target_modules": kwargs.get("target_modules", None)
        }
        
        self.model.setup_for_finetuning("qlora", **lora_config)
        
        trainable, total = self.model.get_trainable_parameters()
        print(f"🔧 Modelo configurado para QLoRA training:")
        print(f"   📊 Parámetros entrenables: {trainable:,}")
        print(f"   📊 Parámetros totales: {total:,}")
        print(f"   📊 Porcentaje entrenable: {trainable/total*100:.1f}%")
        print(f"   🎛️  LoRA rank (r): {self.lora_r}")
        print(f"   🎛️  LoRA alpha: {self.lora_alpha}")
        print(f"   🎛️  LoRA dropout: {self.lora_dropout}")
        print(f"   ⚡ Cuantización activada")
        
    def setup_optimizer(self) -> None:
        """Configura el optimizador para QLoRA training."""
        if self.model.model is None:
            raise RuntimeError("Modelo debe estar cargado antes de configurar optimizador")
            
        # Para QLoRA, usar configuración específica
        qlora_params = []
        for name, param in self.model.model.named_parameters():
            if param.requires_grad:
                qlora_params.append(param)
                
        # Optimizador específico para QLoRA
        self.optimizer = AdamW(
            qlora_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),  # Betas ligeramente diferentes para QLoRA
            eps=1e-6  # Epsilon más pequeño
        )
        
        print(f"✅ Optimizador AdamW configurado para QLoRA:")
        print(f"   📈 Learning rate: {self.learning_rate:.2e}")
        print(f"   ⚖️  Weight decay: {self.weight_decay}")
        print(f"   🎯 Parámetros optimizables: {len(qlora_params)}")
        print(f"   ⚡ Optimización cuantizada activada")
        
    def train_epoch(self, dataloader, epoch: int) -> float:
        """
        Entrena una época con configuraciones específicas para QLoRA.
        """
        # QLoRA puede usar gradient accumulation más agresivo
        gradient_accumulation_steps = 4  # Más pasos para QLoRA
        
        self.model.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            try:
                images = images.to(self.model.device)
                masks = masks.to(self.model.device)
                
                # Forward pass
                outputs = self.model.forward(images)
                loss = self.compute_loss(outputs, masks)
                
                # Escalar loss por gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                # Actualizar cada gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping para QLoRA
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(), 
                        max_norm=1.0  # Clipping estándar para QLoRA
                    )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"  Época {epoch+1} - Batch {batch_idx}: Loss = {loss.item() * gradient_accumulation_steps:.6f}")
                    
            except Exception as e:
                print(f"❌ Error en batch {batch_idx}: {e}")
                continue
                
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
        
    def get_training_summary(self) -> dict:
        """Retorna resumen con parámetros específicos de QLoRA."""
        summary = super().get_training_summary()
        summary["training_method"] = "QLoRATrainer"
        summary["quantization"] = {
            "enabled": True,
            "type": "4-bit",
            "gradient_accumulation_steps": 4
        }
        return summary
