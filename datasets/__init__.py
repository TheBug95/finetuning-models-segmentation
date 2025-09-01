"""
Módulo de datasets para segmentación médica.
Provee datasets limpios y modulares para entrenar modelos SAM.
"""

from .base_dataset import BaseMedicalDataset
from .generic_dataset import GenericCocoOrMaskDataset
from .cataract_dataset import CataractDataset
from .retinopathy_dataset import RetinopathyDataset
from .dataset_factory import create_dataset, list_available_datasets

__all__ = [
    'BaseMedicalDataset',
    'GenericCocoOrMaskDataset',
    'CataractDataset',
    'RetinopathyDataset',
    'create_dataset',
    'list_available_datasets'
]

# Función de compatibilidad con código anterior
def get_dataset(name: str, root: str):
    """Factory function para compatibilidad (deprecated)."""
    import warnings
    warnings.warn("get_dataset está deprecated, usa create_dataset", DeprecationWarning)
    return create_dataset(name, root)
