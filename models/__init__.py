"""
Modelos de segmentación médica para fine-tuning.
Este módulo contiene implementaciones limpias usando Hugging Face Transformers.
"""

from .sam2_model import SAM2Model
from .medsam2_model import MedSAM2Model  
from .mobilesam_model import MobileSAMModel
from .base_model import BaseSegmentationModel

__all__ = [
    'SAM2Model',
    'MedSAM2Model', 
    'MobileSAMModel',
    'BaseSegmentationModel'
]
