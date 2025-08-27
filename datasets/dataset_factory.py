"""
Dataset factory para crear datasets de forma modular.
"""

from typing import Optional, Tuple
from .base_dataset import BaseMedicalDataset
from .cataract_dataset import CataractDataset
from .retinopathy_dataset import RetinopathyDataset


# Registro de datasets disponibles
AVAILABLE_DATASETS = {
    "cataract": CataractDataset,
    "retinopathy": RetinopathyDataset,
    "diabetic_retinopathy": RetinopathyDataset,  # Alias
}


def create_dataset(name: str, 
                  root: str,
                  split: str = "train",
                  image_size: Tuple[int, int] = (512, 512),
                  normalize: bool = True) -> BaseMedicalDataset:
    """
    Factory function para crear datasets.
    
    Args:
        name: Nombre del dataset
        root: Directorio raíz del dataset
        split: Split del dataset ('train', 'val', 'test')
        image_size: Tamaño objetivo de las imágenes
        normalize: Si normalizar las imágenes
        
    Returns:
        Instancia del dataset solicitado
        
    Raises:
        ValueError: Si el dataset no está soportado
    """
    name = name.lower()
    
    if name not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset no soportado: {name}. "
                        f"Disponibles: {list(AVAILABLE_DATASETS.keys())}")
    
    dataset_class = AVAILABLE_DATASETS[name]
    
    return dataset_class(
        root=root,
        split=split,
        image_size=image_size,
        normalize=normalize
    )


def list_available_datasets() -> list:
    """Lista los datasets disponibles."""
    return list(AVAILABLE_DATASETS.keys())


def register_dataset(name: str, dataset_class: type) -> None:
    """
    Registra un nuevo dataset.
    
    Args:
        name: Nombre del dataset
        dataset_class: Clase del dataset (debe heredar de BaseMedicalDataset)
    """
    if not issubclass(dataset_class, BaseMedicalDataset):
        raise ValueError("La clase debe heredar de BaseMedicalDataset")
        
    AVAILABLE_DATASETS[name.lower()] = dataset_class
    print(f"✅ Dataset '{name}' registrado exitosamente")
