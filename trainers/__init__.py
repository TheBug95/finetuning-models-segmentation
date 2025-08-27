"""
Entrenadores modulares para diferentes métodos de fine-tuning.
"""

from .base_trainer import BaseTrainer
from .baseline_trainer import BaselineTrainer
from .lora_trainer import LoRATrainer
from .qlora_trainer import QLoRATrainer

__all__ = [
    'BaseTrainer',
    'BaselineTrainer', 
    'LoRATrainer',
    'QLoRATrainer'
]
