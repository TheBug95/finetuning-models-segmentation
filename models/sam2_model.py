"""
Implementación del modelo SAM2 usando Hugging Face Transformers.
"""

from typing import Optional, List
import torch
from transformers import Sam2Model, Sam2Processor
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
            # Usar float32 por defecto para evitar problemas de tipos mixtos
            self.model = Sam2Model.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float32
            )
            # Almacenar el dtype para conversión posterior de datos
            self.dtype = torch.float32
            print(f"✅ Modelo SAM2 ({self.variant}) cargado desde: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Error cargando modelo SAM2: {e}")
            
    def load_processor(self) -> None:
        """Carga el procesador SAM2."""
        try:
            self.processor = Sam2Processor.from_pretrained(
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
        
    def forward(self, images, input_points=None, input_labels=None, input_boxes=None):
        """
        Forward pass del modelo.
        
        Args:
            images: Tensor de imágenes o lista de imágenes
            input_points: Puntos de entrada opcional
            input_labels: Etiquetas de puntos opcional 
            input_boxes: Bounding boxes opcional
            
        Returns:
            Salida del modelo
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado")
            
        # Convertir inputs al tipo del modelo para evitar incompatibilidad
        model_dtype = getattr(self, 'dtype', torch.float32)
        
        # Procesar inputs si hay procesador disponible
        if self.processor is not None:
            # Preparar argumentos para el procesador
            processor_kwargs = {
                "images": images,
                "return_tensors": "pt"
            }
            
            # Agregar inputs opcionales si están disponibles
            if input_points is not None:
                processor_kwargs["input_points"] = input_points
            if input_labels is not None:
                processor_kwargs["input_labels"] = input_labels
            if input_boxes is not None:
                processor_kwargs["input_boxes"] = input_boxes
                
            inputs = self.processor(**processor_kwargs).to(self.device)
            
            # Convertir al tipo del modelo
            if hasattr(inputs, 'pixel_values'):
                inputs.pixel_values = inputs.pixel_values.to(model_dtype)
            
            # Intentar forward con argumentos completos primero
            try:
                return self.model(**inputs)
            except TypeError as e:
                if "multiple values" in str(e) and "attention_mask" in str(e):
                    # Solo filtrar si hay conflicto confirmado de attention_mask
                    print("⚠️  Detectado conflicto de attention_mask, aplicando filtrado...")
                    filtered_inputs = self._filter_conflicting_args(inputs)
                    return self.model(**filtered_inputs)
                elif "multiple values" in str(e) and "position_ids" in str(e):
                    # Solo filtrar si hay conflicto confirmado de position_ids
                    print("⚠️  Detectado conflicto de position_ids, aplicando filtrado...")
                    filtered_inputs = self._filter_conflicting_args(inputs)
                    return self.model(**filtered_inputs)
                else:
                    # Re-raise otros errores de tipo
                    raise e
        else:
            # Forward directo si no hay procesador (solo pixel_values)
            if hasattr(images, 'to'):
                images = images.to(self.device).to(model_dtype)
            return self.model(pixel_values=images)
            
    def _filter_conflicting_args(self, inputs):
        """
        Filtra argumentos que causan conflictos específicos.
        Solo se llama cuando se detecta un conflicto real.
        """
        # Argumentos esenciales que siempre se mantienen
        essential_args = ['pixel_values', 'original_sizes', 'reshaped_input_sizes']
        # Argumentos problemáticos que se filtran
        problematic_args = ['attention_mask', 'position_ids']
        
        # Si inputs es un objeto con atributos, extraer como diccionario
        if hasattr(inputs, '__dict__'):
            input_dict = {}
            for key in dir(inputs):
                if not key.startswith('_') and not callable(getattr(inputs, key)):
                    input_dict[key] = getattr(inputs, key)
        else:
            input_dict = dict(inputs)
        
        # Filtrar argumentos problemáticos
        filtered_dict = {}
        for key, value in input_dict.items():
            if key in essential_args or key not in problematic_args:
                filtered_dict[key] = value
                
        return filtered_dict
            
    @classmethod
    def list_available_variants(cls) -> dict:
        """Lista las variantes disponibles del modelo."""
        return cls.AVAILABLE_VARIANTS.copy()
