"""
Implementación del modelo MedSAM2 usando Hugging Face Transformers.
"""

from typing import Optional, List
import torch
from transformers import SamModel, SamProcessor, AutoModel
from .base_model import BaseSegmentationModel


class MedSAM2Model(BaseSegmentationModel):
    """Implementación de MedSAM2 usando Hugging Face Transformers."""
    
    # Modelos MedSAM2 disponibles en Hugging Face
    AVAILABLE_VARIANTS = {
        "default": "wanglab/MedSAM2",
        "vit_base": "wanglab/medsam-vit-base",
        "medsam_mix": "guinansu/MedSAMix"
    }
    
    def __init__(self, variant: str = "default", cache_dir: Optional[str] = None):
        """
        Inicializa el modelo MedSAM2.
        
        Args:
            variant: Variante del modelo ('default', 'vit_base', 'medsam_mix')
            cache_dir: Directorio para cache de modelos
        """
        if variant not in self.AVAILABLE_VARIANTS:
            raise ValueError(f"Variante no soportada: {variant}. "
                           f"Disponibles: {list(self.AVAILABLE_VARIANTS.keys())}")
                           
        model_name = self.AVAILABLE_VARIANTS[variant]
        super().__init__(model_name, cache_dir)
        self.variant = variant
        
    def load_model(self) -> None:
        """Carga el modelo MedSAM2 desde Hugging Face."""
        try:
            # Intentar cargar como SamModel primero
            try:
                self.model = SamModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            except Exception:
                # Fallback: cargar como AutoModel genérico
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            
            print(f"✅ Modelo MedSAM2 ({self.variant}) cargado desde: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Error cargando modelo MedSAM2: {e}")
            
    def load_processor(self) -> None:
        """Carga el procesador MedSAM2."""
        try:
            # Intentar cargar procesador específico
            try:
                self.processor = SamProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
            except Exception:
                # Fallback: usar procesador SAM genérico
                self.processor = SamProcessor.from_pretrained(
                    "facebook/sam-vit-base",
                    cache_dir=self.cache_dir
                )
                print("⚠️  Usando procesador SAM genérico para MedSAM2")
            
            print(f"✅ Procesador MedSAM2 cargado")
        except Exception as e:
            print(f"⚠️  No se pudo cargar procesador MedSAM2: {e}")
            self.processor = None
            
    def _get_default_lora_targets(self) -> List[str]:
        """Retorna los módulos objetivo por defecto para LoRA en MedSAM2."""
        # Targets similares a SAM pero adaptados para MedSAM2
        return [
            "vision_encoder.patch_embed.proj",
            "vision_encoder.blocks.0.attn.qkv", 
            "vision_encoder.blocks.0.attn.proj",
            "vision_encoder.blocks.1.attn.qkv",
            "vision_encoder.blocks.1.attn.proj",
            "mask_decoder.transformer.layers.0.self_attn.q_proj",
            "mask_decoder.transformer.layers.0.self_attn.k_proj",
            "mask_decoder.transformer.layers.0.self_attn.v_proj"
        ]
        
    def forward(self, images, input_points=None, input_labels=None):
        """
        Forward pass del modelo MedSAM2.
        
        Args:
            images: Tensor de imágenes
            input_points: Puntos de entrada opcional
            input_labels: Etiquetas de puntos opcional
            
        Returns:
            Salida del modelo
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado")
            
        # Procesar inputs si hay procesador disponible
        if self.processor is not None:
            inputs = self.processor(
                images=images,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt"
            ).to(self.device)
            return self.model(**inputs)
        else:
            # Forward directo si no hay procesador
            if hasattr(self.model, 'forward'):
                return self.model(images)
            else:
                # Para modelos que no tienen forward estándar
                return self.model(pixel_values=images)
                
    @classmethod
    def list_available_variants(cls) -> dict:
        """Lista las variantes disponibles del modelo."""
        return cls.AVAILABLE_VARIANTS.copy()
