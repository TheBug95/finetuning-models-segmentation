"""
Dataset base para segmentación médica.
Define la interfaz común y funcionalidades compartidas.
"""

import os
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

try:
    from pycocotools.coco import COCO
    COCO_AVAILABLE = True
except ImportError:
    COCO_AVAILABLE = False


class BaseMedicalDataset(Dataset, ABC):
    """Clase base para datasets de segmentación médica."""
    
    def __init__(self, 
                 root: str,
                 split: str = "train",
                 image_size: Tuple[int, int] = (512, 512),
                 normalize: bool = True):
        """
        Inicializa el dataset base.
        
        Args:
            root: Directorio raíz del dataset
            split: Split del dataset ('train', 'val', 'test')
            image_size: Tamaño objetivo de las imágenes
            normalize: Si normalizar las imágenes
        """
        self.root = root
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        
        # Verificar que el directorio existe
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset root no encontrado: {root}")
            
        # Configurar transformaciones
        self.image_transform = self._setup_image_transforms()
        self.mask_transform = self._setup_mask_transforms()
        
        # Cargar datos
        self.data_list = self._load_data_list()
        
        print(f"✅ Dataset cargado: {len(self.data_list)} muestras en split '{split}'")
        
    def _setup_image_transforms(self) -> T.Compose:
        """Configura las transformaciones para imágenes."""
        transforms = [
            T.Resize(self.image_size),
            T.ToTensor()
        ]
        
        if self.normalize:
            # Normalización ImageNet por defecto
            transforms.append(T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
            
        return T.Compose(transforms)
        
    def _setup_mask_transforms(self) -> T.Compose:
        """Configura las transformaciones para máscaras."""
        return T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])
        
    @abstractmethod
    def _load_data_list(self) -> List[Dict[str, Any]]:
        """Carga la lista de datos del dataset."""
        pass
        
    @abstractmethod
    def _load_image(self, image_path: str) -> Image.Image:
        """Carga una imagen del dataset."""
        pass
        
    @abstractmethod
    def _load_mask(self, mask_path: str) -> Image.Image:
        """Carga una máscara del dataset."""
        pass
        
    def __len__(self) -> int:
        """Retorna el tamaño del dataset."""
        return len(self.data_list)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtiene un elemento del dataset.
        
        Args:
            idx: Índice del elemento
            
        Returns:
            Tuple de (imagen, máscara) como tensores
        """
        data_item = self.data_list[idx]
        
        # Cargar imagen y máscara
        image = self._load_image(data_item["image_path"])
        mask = self._load_mask(data_item["mask_path"])
        
        # Aplicar transformaciones
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            
        return image, mask
        
    def get_class_distribution(self) -> Dict[str, int]:
        """Retorna la distribución de clases en el dataset."""
        # Implementación básica - puede ser sobrescrita
        return {"samples": len(self.data_list)}
        
    def get_dataset_info(self) -> Dict[str, Any]:
        """Retorna información del dataset."""
        return {
            "name": self.__class__.__name__,
            "root": self.root,
            "split": self.split,
            "size": len(self.data_list),
            "image_size": self.image_size,
            "normalize": self.normalize,
            "class_distribution": self.get_class_distribution()
        }
        
    @staticmethod
    def _find_images_and_masks(root: str, 
                              image_extensions: List[str] = None,
                              mask_extensions: List[str] = None) -> List[Dict[str, str]]:
        """
        Busca pares imagen-máscara en el directorio.
        
        Args:
            root: Directorio donde buscar
            image_extensions: Extensiones válidas para imágenes
            mask_extensions: Extensiones válidas para máscaras
            
        Returns:
            Lista de diccionarios con paths de imagen y máscara
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        if mask_extensions is None:
            mask_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            
        # Buscar estructura images/masks
        images_dir = os.path.join(root, "images")
        masks_dir = os.path.join(root, "masks")
        
        data_list = []
        
        if os.path.exists(images_dir) and os.path.exists(masks_dir):
            # Estructura con directorios separados
            for img_file in os.listdir(images_dir):
                if any(img_file.lower().endswith(ext) for ext in image_extensions):
                    # Buscar máscara correspondiente
                    base_name = os.path.splitext(img_file)[0]
                    
                    for mask_ext in mask_extensions:
                        mask_file = base_name + mask_ext
                        mask_path = os.path.join(masks_dir, mask_file)
                        
                        if os.path.exists(mask_path):
                            data_list.append({
                                "image_path": os.path.join(images_dir, img_file),
                                "mask_path": mask_path
                            })
                            break
        else:
            # Buscar en el directorio raíz
            for file in os.listdir(root):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    if not any(keyword in file.lower() for keyword in ['mask', 'label', 'gt']):
                        # Es una imagen, buscar su máscara
                        base_name = os.path.splitext(file)[0]
                        
                        # Patrones comunes para máscaras
                        mask_patterns = [
                            base_name + "_mask",
                            base_name + "_label", 
                            base_name + "_gt",
                            "mask_" + base_name,
                            "label_" + base_name
                        ]
                        
                        for pattern in mask_patterns:
                            for mask_ext in mask_extensions:
                                mask_file = pattern + mask_ext
                                mask_path = os.path.join(root, mask_file)
                                
                                if os.path.exists(mask_path):
                                    data_list.append({
                                        "image_path": os.path.join(root, file),
                                        "mask_path": mask_path
                                    })
                                    break
                            else:
                                continue
                            break
                            
        return data_list
