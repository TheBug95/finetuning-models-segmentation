"""
Implementación del modelo SAM2 usando Hugging Face Transformers.
"""

from typing import Optional, List
import torch
from transformers import SamModel, SamProcessor
from .base_model import BaseSegmentationModel


class SAM2Model(BaseSegmentationModel):
    """Implementación de SAM2 usando Hugging Face Transformers."""
    
    # Modelos SAM2 disponibles en Hugging Face
    AVAILABLE_VARIANTS = {
        "tiny": "facebook/sam2-hiera-tiny",
        "base": "facebook/sam2-hiera-base-plus", 
        "large": "facebook/sam2-hiera-large",
        "huge": "facebook/sam2.1-hiera-large"
    }
    
    def __init__(self, variant: str = "tiny", cache_dir: Optional[str] = None):
        """
        Inicializa el modelo SAM2.
        
        Args:
            variant: Variante del modelo ('tiny', 'base', 'large', 'huge')
            cache_dir: Directorio para cache de modelos
        """
        if variant not in self.AVAILABLE_VARIANTS:
            raise ValueError(f"Variante no soportada: {variant}. "
                           f"Disponibles: {list(self.AVAILABLE_VARIANTS.keys())}")
                           
        model_name = self.AVAILABLE_VARIANTS[variant]
        super().__init__(model_name, cache_dir)
        self.variant = variant
        
    def load_model(self) -> None:
        """Carga el modelo SAM2 desde Hugging Face."""
        try:
            self.model = SamModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            print(f"✅ Modelo SAM2 ({self.variant}) cargado desde: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Error cargando modelo SAM2: {e}")
            
    def load_processor(self) -> None:
        """Carga el procesador SAM2."""
        try:
            self.processor = SamProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            print(f"✅ Procesador SAM2 cargado")
        except Exception as e:
            raise RuntimeError(f"Error cargando procesador SAM2: {e}")
            
    def _get_default_lora_targets(self) -> List[str]:
        """Retorna los módulos objetivo por defecto para LoRA en SAM2."""
        return [
            "vision_encoder.patch_embed.proj",
            "vision_encoder.blocks.0.attn.qkv",
            "vision_encoder.blocks.0.attn.proj",
            "mask_decoder.transformer.layers.0.self_attn.q_proj",
            "mask_decoder.transformer.layers.0.self_attn.k_proj",
            "mask_decoder.transformer.layers.0.self_attn.v_proj",
            "mask_decoder.transformer.layers.0.self_attn.out_proj"
        ]
        
    def forward(self, images, input_points=None, input_labels=None):
        """
        Forward pass del modelo.
        
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
            return self.model(images)
            
    @classmethod
    def list_available_variants(cls) -> dict:
        """Lista las variantes disponibles del modelo."""
        return cls.AVAILABLE_VARIANTS.copy()
