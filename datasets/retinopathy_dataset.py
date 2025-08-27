"""
Dataset para segmentación de retinopatía diabética.
"""

import os
import json
from typing import List, Dict, Any
from PIL import Image
from .base_dataset import BaseMedicalDataset, COCO_AVAILABLE


class RetinopathyDataset(BaseMedicalDataset):
    """Dataset especializado para segmentación de retinopatía diabética."""
    
    def __init__(self, root: str, split: str = "train", **kwargs):
        """
        Inicializa el dataset de retinopatía.
        
        Args:
            root: Directorio raíz del dataset
            split: Split del dataset ('train', 'val', 'test')
            **kwargs: Argumentos adicionales para BaseMedicalDataset
        """
        self.coco_mode = False
        super().__init__(root, split, **kwargs)
        
    def _load_data_list(self) -> List[Dict[str, Any]]:
        """Carga la lista de datos del dataset de retinopatía."""
        # Primero intentar formato COCO
        coco_file = os.path.join(self.root, "_annotations.coco.json")
        
        if os.path.exists(coco_file) and COCO_AVAILABLE:
            return self._load_coco_data(coco_file)
        else:
            return self._load_standard_data()
            
    def _load_coco_data(self, coco_file: str) -> List[Dict[str, Any]]:
        """Carga datos en formato COCO."""
        try:
            from pycocotools.coco import COCO
            
            coco = COCO(coco_file)
            self.coco_mode = True
            
            data_list = []
            
            # Obtener todas las imágenes
            img_ids = coco.getImgIds()
            
            for img_id in img_ids:
                img_info = coco.loadImgs(img_id)[0]
                image_path = os.path.join(self.root, img_info['file_name'])
                
                if not os.path.exists(image_path):
                    continue
                    
                # Obtener anotaciones para esta imagen
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                
                if anns:  # Solo si hay anotaciones
                    data_list.append({
                        "image_path": image_path,
                        "mask_path": None,  # Para COCO generamos la máscara on-the-fly
                        "annotations": anns,
                        "image_info": img_info,
                        "coco": coco
                    })
                    
            print(f"✅ Datos COCO cargados: {len(data_list)} imágenes con anotaciones")
            return data_list
            
        except Exception as e:
            print(f"⚠️  Error cargando COCO: {e}. Usando formato estándar.")
            return self._load_standard_data()
            
    def _load_standard_data(self) -> List[Dict[str, Any]]:
        """Carga datos en formato estándar (imágenes + máscaras)."""
        split_dir = os.path.join(self.root, self.split)
        
        if os.path.exists(split_dir):
            # Estructura con splits separados
            data_list = self._find_images_and_masks(split_dir)
        else:
            # Estructura sin splits
            data_list = self._find_images_and_masks(self.root)
            
        # Convertir a formato estándar
        formatted_data = []
        for item in data_list:
            formatted_data.append({
                "image_path": item["image_path"],
                "mask_path": item["mask_path"],
                "annotations": None,
                "image_info": None,
                "coco": None
            })
            
        print(f"✅ Datos estándar cargados: {len(formatted_data)} pares imagen-máscara")
        return formatted_data
        
    def _load_image(self, image_path: str) -> Image.Image:
        """Carga una imagen del dataset."""
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            raise RuntimeError(f"Error cargando imagen {image_path}: {e}")
            
    def _load_mask(self, mask_path: str = None, data_item: Dict = None) -> Image.Image:
        """Carga una máscara del dataset."""
        if self.coco_mode and data_item:
            # Generar máscara desde anotaciones COCO
            return self._generate_coco_mask(data_item)
        elif mask_path and os.path.exists(mask_path):
            # Cargar máscara desde archivo
            try:
                mask = Image.open(mask_path).convert("L")  # Grayscale
                return mask
            except Exception as e:
                raise RuntimeError(f"Error cargando máscara {mask_path}: {e}")
        else:
            raise RuntimeError("No se puede cargar la máscara")
            
    def _generate_coco_mask(self, data_item: Dict) -> Image.Image:
        """Genera máscara desde anotaciones COCO."""
        try:
            import numpy as np
            from pycocotools import mask as maskUtils
            
            coco = data_item["coco"]
            anns = data_item["annotations"]
            img_info = data_item["image_info"]
            
            # Crear máscara combinada
            combined_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            
            for ann in anns:
                # Convertir anotación a máscara
                if 'segmentation' in ann:
                    mask = coco.annToMask(ann)
                    combined_mask = np.maximum(combined_mask, mask * 255)
                    
            return Image.fromarray(combined_mask, mode='L')
            
        except Exception as e:
            raise RuntimeError(f"Error generando máscara COCO: {e}")
            
    def __getitem__(self, idx: int):
        """Override para manejar el modo COCO."""
        data_item = self.data_list[idx]
        
        # Cargar imagen
        image = self._load_image(data_item["image_path"])
        
        # Cargar máscara
        if self.coco_mode:
            mask = self._load_mask(data_item=data_item)
        else:
            mask = self._load_mask(data_item["mask_path"])
            
        # Aplicar transformaciones
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            
        return image, mask
        
    def get_class_distribution(self) -> Dict[str, int]:
        """Retorna la distribución de clases en el dataset de retinopatía."""
        return {
            "total_samples": len(self.data_list),
            "format": "COCO" if self.coco_mode else "Standard",
            "split": self.split,
            "disease_type": "diabetic_retinopathy"
        }
