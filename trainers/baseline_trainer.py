"""
Entrenador para fine-tuning baseline (completo).
"""

import torch
from torch.optim import AdamW
from .base_trainer import BaseTrainer


class BaselineTrainer(BaseTrainer):
    """Entrenador para fine-tuning baseline (todas las capas)."""
    
    def __init__(self, model, learning_rate: float = 1e-5, weight_decay: float = 0.01):
        """
        Inicializa el entrenador baseline.
        
        Args:
            model: Modelo de segmentación
            learning_rate: LR más bajo para baseline (fine-tuning completo)
            weight_decay: Decay de pesos
        """
        # LR más bajo por defecto para baseline
        super().__init__(model, learning_rate, weight_decay)
        
    def setup_model_for_training(self, **kwargs) -> None:
        """Configura el modelo para fine-tuning baseline."""
        self.model.setup_for_finetuning("baseline", **kwargs)
        
        trainable, total = self.model.get_trainable_parameters()
        print(f"🔧 Modelo configurado para BASELINE training:")
        print(f"   📊 Parámetros entrenables: {trainable:,}")
        print(f"   📊 Parámetros totales: {total:,}")
        print(f"   📊 Porcentaje entrenable: {trainable/total*100:.1f}%")
        
    def setup_optimizer(self) -> None:
        """Configura el optimizador para baseline training."""
        if self.model.model is None:
            raise RuntimeError("Modelo debe estar cargado antes de configurar optimizador")
            
        # Usar AdamW con parámetros más conservadores para baseline
        self.optimizer = AdamW(
            self.model.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        print(f"✅ Optimizador AdamW configurado para baseline:")
        print(f"   📈 Learning rate: {self.learning_rate:.2e}")
        print(f"   ⚖️  Weight decay: {self.weight_decay}")
        
    def train_epoch(self, dataloader, epoch: int) -> float:
        """
        Entrena una época con configuraciones específicas para baseline.
        """
        # Aplicar gradient accumulation para baseline si es necesario
        gradient_accumulation_steps = 2  # Acumular gradientes para estabilidad
        
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
                    # Gradient clipping más agresivo para baseline
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(), 
                        max_norm=0.5  # Más agresivo para baseline
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
