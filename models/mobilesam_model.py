"""
Implementación del modelo MobileSAM usando Hugging Face Transformers.
"""

from typing import Optional, List
import torch
from transformers import SamModel, SamProcessor, AutoModel
from .base_model import BaseSegmentationModel


class MobileSAMModel(BaseSegmentationModel):
    """Implementación de MobileSAM usando Hugging Face Transformers."""
    
    # Modelos MobileSAM disponibles en Hugging Face
    AVAILABLE_VARIANTS = {
        "default": "nielsr/mobilesam",
        "qualcomm": "qualcomm/MobileSam", 
        "dhkim": "dhkim2810/MobileSAM",
        "v2": "RogerQi/MobileSAMV2"
    }
    
    def __init__(self, variant: str = "default", cache_dir: Optional[str] = None):
        """
        Inicializa el modelo MobileSAM.
        
        Args:
            variant: Variante del modelo ('default', 'qualcomm', 'dhkim', 'v2')
            cache_dir: Directorio para cache de modelos
        """
        if variant not in self.AVAILABLE_VARIANTS:
            raise ValueError(f"Variante no soportada: {variant}. "
                           f"Disponibles: {list(self.AVAILABLE_VARIANTS.keys())}")
                           
        model_name = self.AVAILABLE_VARIANTS[variant]
        super().__init__(model_name, cache_dir)
        self.variant = variant
        
    def load_model(self) -> None:
        """Carga el modelo MobileSAM desde Hugging Face."""
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
            
            print(f"✅ Modelo MobileSAM ({self.variant}) cargado desde: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Error cargando modelo MobileSAM: {e}")
            
    def load_processor(self) -> None:
        """Carga el procesador MobileSAM."""
        try:
            # Intentar cargar procesador específico
            try:
                self.processor = SamProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
            except Exception:
                # Fallback: usar procesador SAM genérico (más compatible con MobileSAM)
                self.processor = SamProcessor.from_pretrained(
                    "facebook/sam-vit-base",
                    cache_dir=self.cache_dir
                )
                print("⚠️  Usando procesador SAM genérico para MobileSAM")
            
            print(f"✅ Procesador MobileSAM cargado")
        except Exception as e:
            print(f"⚠️  No se pudo cargar procesador MobileSAM: {e}")
            self.processor = None
            
    def _get_default_lora_targets(self) -> List[str]:
        """Retorna los módulos objetivo por defecto para LoRA en MobileSAM."""
        # Targets optimizados para MobileSAM (modelo más ligero)
        return [
            "vision_encoder.patch_embed.proj",
            "vision_encoder.blocks.0.attn.qkv",
            "vision_encoder.blocks.0.attn.proj",
            "vision_encoder.blocks.1.attn.qkv",
            "vision_encoder.blocks.1.attn.proj", 
            "mask_decoder.transformer.layers.0.self_attn.q_proj",
            "mask_decoder.transformer.layers.0.self_attn.v_proj"
        ]
        
    def forward(self, images, input_points=None, input_labels=None):
        """
        Forward pass del modelo MobileSAM.
        
        Args:
            images: Tensor de imágenes
            input_points: Puntos de entrada opcional
            input_labels: Etiquetas de puntos opcional
            
        Returns:
            Salida del modelo
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado")
            
        # Convertir inputs al tipo del modelo para evitar incompatibilidad
        model_dtype = getattr(self, 'dtype', torch.float32)
        
        # Procesar inputs si hay procesador disponible
        if self.processor is not None:
            inputs = self.processor(
                images=images,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt"
            ).to(self.device)
            
            # Convertir al tipo del modelo
            if hasattr(inputs, 'pixel_values'):
                inputs.pixel_values = inputs.pixel_values.to(model_dtype)
            
            # Filtrar argumentos problemáticos que causan conflictos
            filtered_inputs = {}
            for key, value in inputs.items():
                if key in ['pixel_values', 'original_sizes', 'reshaped_input_sizes']:
                    filtered_inputs[key] = value
                elif key not in ['attention_mask', 'position_ids']:
                    filtered_inputs[key] = value
            
            # Intentar forward con argumentos completos primero
            try:
                return self.model(**inputs)
            except TypeError as e:
                if "multiple values" in str(e) and ("attention_mask" in str(e) or "position_ids" in str(e)):
                    # Solo filtrar si hay conflicto confirmado
                    print("⚠️  Detectado conflicto de argumentos, aplicando filtrado...")
                    return self.model(**filtered_inputs)
                else:
                    # Re-raise otros errores de tipo
                    raise e
        else:
            # Forward directo si no hay procesador
            if hasattr(images, 'to'):
                images = images.to(self.device).to(model_dtype)
            return self.model(pixel_values=images)
                
    def optimize_for_mobile(self) -> None:
        """Optimizaciones específicas para dispositivos móviles."""
        if self.model is None:
            raise RuntimeError("Modelo no cargado")
            
        # Aplicar optimizaciones móviles
        self.model.eval()
        
        # Convertir a half precision si es posible
        if torch.cuda.is_available():
            self.model = self.model.half()
            
        print("✅ Modelo optimizado para dispositivos móviles")
        
    @classmethod
    def list_available_variants(cls) -> dict:
        """Lista las variantes disponibles del modelo."""
        return cls.AVAILABLE_VARIANTS.copy()
