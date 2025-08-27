"""
Clase base para todos los entrenadores.
Define la interfaz común y funcionalidades compartidas.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import time
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from models.base_model import BaseSegmentationModel


class TrainingMetrics:
    """Clase para tracking de métricas de entrenamiento."""
    
    def __init__(self):
        self.epoch_losses: List[float] = []
        self.epoch_times: List[float] = []
        self.learning_rates: List[float] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
    def start_training(self) -> None:
        """Marca el inicio del entrenamiento."""
        self.start_time = time.time()
        
    def end_training(self) -> None:
        """Marca el final del entrenamiento."""
        self.end_time = time.time()
        
    def add_epoch(self, loss: float, epoch_time: float, lr: float) -> None:
        """Añade métricas de una época."""
        self.epoch_losses.append(loss)
        self.epoch_times.append(epoch_time)
        self.learning_rates.append(lr)
        
    @property
    def total_time(self) -> float:
        """Tiempo total de entrenamiento."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
        
    @property
    def best_loss(self) -> float:
        """Mejor loss obtenido."""
        return min(self.epoch_losses) if self.epoch_losses else float('inf')
        
    @property
    def improvement(self) -> float:
        """Mejora desde la primera hasta la última época."""
        if len(self.epoch_losses) < 2:
            return 0.0
        return self.epoch_losses[0] - self.epoch_losses[-1]
        
    def to_dict(self) -> Dict[str, Any]:
        """Convierte las métricas a diccionario."""
        return {
            "epoch_losses": self.epoch_losses,
            "epoch_times": self.epoch_times,
            "learning_rates": self.learning_rates,
            "total_time": self.total_time,
            "best_loss": self.best_loss,
            "improvement": self.improvement,
            "final_loss": self.epoch_losses[-1] if self.epoch_losses else None
        }


class BaseTrainer(ABC):
    """Clase base abstracta para entrenadores."""
    
    def __init__(self, 
                 model: BaseSegmentationModel,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01):
        """
        Inicializa el entrenador base.
        
        Args:
            model: Modelo de segmentación
            learning_rate: Tasa de aprendizaje
            weight_decay: Decay de pesos
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
        self.metrics = TrainingMetrics()
        
    @abstractmethod
    def setup_optimizer(self) -> None:
        """Configura el optimizador específico del método."""
        pass
        
    @abstractmethod
    def setup_model_for_training(self, **kwargs) -> None:
        """Configura el modelo para el método específico de entrenamiento."""
        pass
        
    def setup_scheduler(self, scheduler_type: str = "cosine", **kwargs) -> None:
        """Configura el scheduler de learning rate."""
        if self.optimizer is None:
            raise RuntimeError("Optimizador debe configurarse antes que el scheduler")
            
        if scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            T_max = kwargs.get("T_max", 10)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_type == "step":
            from torch.optim.lr_scheduler import StepLR
            step_size = kwargs.get("step_size", 3)
            gamma = kwargs.get("gamma", 0.1)
            self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == "reduce_on_plateau":
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3)
        else:
            print(f"⚠️  Scheduler desconocido: {scheduler_type}. No se usará scheduler.")
            
    def compute_loss(self, outputs, targets) -> torch.Tensor:
        """
        Calcula el loss entre outputs y targets.
        
        Args:
            outputs: Salidas del modelo
            targets: Targets ground truth
            
        Returns:
            Loss calculado
        """
        # Loss genérico - puede ser sobrescrito por subclases
        if hasattr(outputs, 'loss'):
            return outputs.loss
        
        # Fallback: usar MSE loss
        if isinstance(outputs, dict):
            pred_masks = outputs.get('pred_masks', outputs.get('masks', outputs))
        else:
            pred_masks = outputs
            
        loss_fn = nn.MSELoss()
        return loss_fn(pred_masks, targets)
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """
        Entrena una época.
        
        Args:
            dataloader: DataLoader con los datos
            epoch: Número de época actual
            
        Returns:
            Loss promedio de la época
        """
        self.model.model.train()
        total_loss = 0.0
        num_batches = 0
        
        epoch_start = time.time()
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            try:
                # Mover datos al dispositivo
                images = images.to(self.model.device)
                masks = masks.to(self.model.device)
                
                # Forward pass
                outputs = self.model.forward(images)
                
                # Calcular loss
                loss = self.compute_loss(outputs, masks)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
                
                # Optimización
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Log cada 10 batches
                if batch_idx % 10 == 0:
                    print(f"  Época {epoch+1} - Batch {batch_idx}: Loss = {loss.item():.6f}")
                    
            except Exception as e:
                print(f"❌ Error en batch {batch_idx}: {e}")
                continue
                
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Actualizar scheduler si existe
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()
                
        # Registrar métricas
        current_lr = self.optimizer.param_groups[0]['lr']
        self.metrics.add_epoch(avg_loss, epoch_time, current_lr)
        
        return avg_loss
        
    def train(self, 
              dataloader: DataLoader,
              epochs: int,
              output_dir: Optional[str] = None,
              save_every: int = 5) -> TrainingMetrics:
        """
        Ejecuta el entrenamiento completo.
        
        Args:
            dataloader: DataLoader con los datos
            epochs: Número de épocas
            output_dir: Directorio para guardar checkpoints
            save_every: Guardar checkpoint cada N épocas
            
        Returns:
            Métricas de entrenamiento
        """
        print(f"🚀 Iniciando entrenamiento por {epochs} épocas")
        print("="*60)
        
        self.metrics.start_training()
        
        for epoch in range(epochs):
            print(f"\n📈 Época {epoch + 1}/{epochs}")
            print("-" * 40)
            
            avg_loss = self.train_epoch(dataloader, epoch)
            
            print(f"✅ Época {epoch + 1} completada:")
            print(f"   Loss promedio: {avg_loss:.6f}")
            print(f"   Tiempo: {self.metrics.epoch_times[-1]:.2f}s")
            print(f"   Learning rate: {self.metrics.learning_rates[-1]:.2e}")
            
            # Guardar checkpoint periódicamente
            if output_dir and (epoch + 1) % save_every == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}")
                self.save_checkpoint(checkpoint_dir, epoch)
                
        self.metrics.end_training()
        
        print(f"\n🎯 ENTRENAMIENTO COMPLETADO")
        print("="*60)
        print(f"⏱️  Tiempo total: {self.metrics.total_time:.2f}s")
        print(f"📈 Mejor loss: {self.metrics.best_loss:.6f}")
        print(f"📈 Mejora total: {self.metrics.improvement:.6f}")
        
        return self.metrics
        
    def save_checkpoint(self, output_dir: str, epoch: int) -> None:
        """Guarda un checkpoint del entrenamiento."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar modelo
        self.model.save_model(output_dir)
        
        # Guardar estado del optimizador
        torch.save(self.optimizer.state_dict(), 
                  os.path.join(output_dir, "optimizer.pt"))
        
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(),
                      os.path.join(output_dir, "scheduler.pt"))
                      
        # Guardar métricas
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
            
        # Guardar info del checkpoint
        checkpoint_info = {
            "epoch": epoch,
            "model_info": self.model.info,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay
        }
        
        with open(os.path.join(output_dir, "checkpoint_info.json"), "w") as f:
            json.dump(checkpoint_info, f, indent=2)
            
        print(f"💾 Checkpoint guardado en: {output_dir}")
        
    def get_training_summary(self) -> Dict[str, Any]:
        """Retorna un resumen del entrenamiento."""
        trainable, total = self.model.get_trainable_parameters()
        
        return {
            "model_name": self.model.model_name,
            "training_method": self.__class__.__name__,
            "parameters": {
                "trainable": trainable,
                "total": total,
                "percentage": (trainable / total * 100) if total > 0 else 0
            },
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay
            },
            "metrics": self.metrics.to_dict()
        }
