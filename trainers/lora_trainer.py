"""
Entrenador para fine-tuning con LoRA.
"""

import torch
from torch.optim import AdamW
from .base_trainer import BaseTrainer


class LoRATrainer(BaseTrainer):
    """Entrenador para fine-tuning con LoRA."""
    
    def __init__(self, 
                 model, 
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1):
        """
        Inicializa el entrenador LoRA.
        
        Args:
            model: Modelo de segmentación
            learning_rate: Tasa de aprendizaje
            weight_decay: Decay de pesos
            lora_r: Rango de LoRA
            lora_alpha: Alpha de LoRA  
            lora_dropout: Dropout de LoRA
        """
        super().__init__(model, learning_rate, weight_decay)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
    def setup_model_for_training(self, **kwargs) -> None:
        """Configura el modelo para fine-tuning con LoRA."""
        lora_config = {
            "r": kwargs.get("r", self.lora_r),
            "lora_alpha": kwargs.get("lora_alpha", self.lora_alpha),
            "lora_dropout": kwargs.get("lora_dropout", self.lora_dropout),
            "target_modules": kwargs.get("target_modules", None)
        }
        
        self.model.setup_for_finetuning("lora", **lora_config)
        
        trainable, total = self.model.get_trainable_parameters()
        print(f"🔧 Modelo configurado para LoRA training:")
        print(f"   📊 Parámetros entrenables: {trainable:,}")
        print(f"   📊 Parámetros totales: {total:,}")
        print(f"   📊 Porcentaje entrenable: {trainable/total*100:.1f}%")
        print(f"   🎛️  LoRA rank (r): {self.lora_r}")
        print(f"   🎛️  LoRA alpha: {self.lora_alpha}")
        print(f"   🎛️  LoRA dropout: {self.lora_dropout}")
        
    def setup_optimizer(self) -> None:
        """Configura el optimizador para LoRA training."""
        if self.model.model is None:
            raise RuntimeError("Modelo debe estar cargado antes de configurar optimizador")
            
        # Solo optimizar parámetros LoRA
        lora_params = []
        for name, param in self.model.model.named_parameters():
            if param.requires_grad:
                lora_params.append(param)
                
        self.optimizer = AdamW(
            lora_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        print(f"✅ Optimizador AdamW configurado para LoRA:")
        print(f"   📈 Learning rate: {self.learning_rate:.2e}")
        print(f"   ⚖️  Weight decay: {self.weight_decay}")
        print(f"   🎯 Parámetros optimizables: {len(lora_params)}")
        
    def compute_loss(self, outputs, targets) -> torch.Tensor:
        """Calcula el loss con regularización específica para LoRA."""
        base_loss = super().compute_loss(outputs, targets)
        
        # Añadir regularización LoRA si es necesario
        # Por ahora usar el loss base
        return base_loss
        
    def get_training_summary(self) -> dict:
        """Retorna resumen con parámetros específicos de LoRA."""
        summary = super().get_training_summary()
        summary["lora_config"] = {
            "r": self.lora_r,
            "alpha": self.lora_alpha,
            "dropout": self.lora_dropout
        }
        return summary
