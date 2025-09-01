"""
Clase base para todos los modelos de segmentación.
Define la interfaz común y métodos compartidos.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel, BitsAndBytesConfig

# Import PEFT con manejo de errores para compatibilidad
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  PEFT no disponible. LoRA/QLoRA no funcionarán.")


class BaseSegmentationModel(ABC):
    """Clase base abstracta para modelos de segmentación médica."""
    
    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        """
        Inicializa el modelo base.
        
        Args:
            model_name: Nombre del modelo en Hugging Face
            cache_dir: Directorio para cache de modelos
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model: Optional[PreTrainedModel] = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32  # Tipo por defecto
        
    @abstractmethod
    def load_model(self) -> None:
        """Carga el modelo desde Hugging Face."""
        pass
        
    @abstractmethod
    def load_processor(self) -> None:
        """Carga el procesador asociado al modelo."""
        pass
        
    def setup_for_finetuning(self, method: str, **kwargs) -> None:
        """
        Configura el modelo para fine-tuning.
        
        Args:
            method: Método de fine-tuning ('baseline', 'lora', 'qlora')
            **kwargs: Parámetros adicionales específicos del método
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado. Llama a load_model() primero.")
            
        if method == "baseline":
            self._setup_baseline_training()
        elif method == "lora":
            self._setup_lora_training(**kwargs)
        elif method == "qlora":
            # Recargar el modelo en 4-bit usando BitsAndBytes
            if self.model is not None:
                model_cls = self.model.__class__
            else:
                raise RuntimeError("Modelo no cargado. Llama a load_model() primero.")

            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            self.model = model_cls.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                quantization_config=bnb_config,
                load_in_4bit=True,
                device_map="auto",
            )
            # Actualizar dtype para inputs
            self.dtype = getattr(bnb_config, "bnb_4bit_compute_dtype", torch.float16)

            # Preparar modelo para entrenamiento en k-bits
            if PEFT_AVAILABLE:
                self.model = prepare_model_for_kbit_training(self.model)

            # Aplicar configuración LoRA sobre el modelo cuantizado
            self._setup_lora_training(**kwargs)
            print(f"✅ Modelo configurado para fine-tuning con QLoRA")
        else:
            raise ValueError(f"Método no soportado: {method}")
            
    def _setup_baseline_training(self) -> None:
        """Configura para fine-tuning completo (baseline)."""
        for param in self.model.parameters():
            param.requires_grad = True
        print(f"✅ Modelo configurado para fine-tuning baseline")
        
    def _setup_lora_training(self, 
                           r: int = 16, 
                           lora_alpha: int = 32, 
                           lora_dropout: float = 0.1,
                           target_modules: Optional[list] = None) -> None:
        """Configura para fine-tuning con LoRA."""
        if not PEFT_AVAILABLE:
            print("❌ PEFT no disponible. Aplicando congelamiento manual...")
            self._setup_manual_freezing()
            return
            
        if target_modules is None:
            target_modules = self._get_default_lora_targets()
            
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print(f"✅ Modelo configurado para fine-tuning con LoRA")
        
    def _setup_manual_freezing(self) -> None:
        """Congelamiento manual cuando PEFT no está disponible."""
        # Congelar todo excepto las últimas capas
        total_params = list(self.model.parameters())
        for param in total_params[:-10]:  # Descongelar últimas 10 capas
            param.requires_grad = False
        print("✅ Congelamiento manual aplicado (últimas 10 capas entrenables)")
        
    @abstractmethod
    def _get_default_lora_targets(self) -> list:
        """Retorna los módulos objetivo por defecto para LoRA."""
        pass
        
    def get_trainable_parameters(self) -> Tuple[int, int]:
        """
        Retorna el número de parámetros entrenables y totales.
        
        Returns:
            Tuple con (parámetros_entrenables, parámetros_totales)
        """
        if self.model is None:
            return 0, 0
            
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total
        
    def save_model(self, output_dir: str) -> None:
        """Guarda el modelo y procesador."""
        if self.model is None:
            raise RuntimeError("No hay modelo para guardar")
            
        self.model.save_pretrained(output_dir)
        if self.processor is not None:
            self.processor.save_pretrained(output_dir)
        print(f"✅ Modelo guardado en: {output_dir}")
        
    def to_device(self) -> None:
        """Mueve el modelo al dispositivo configurado."""
        if self.model is not None:
            self.model.to(self.device)
            
    @property
    def info(self) -> Dict[str, Any]:
        """Retorna información del modelo."""
        trainable, total = self.get_trainable_parameters()
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_percentage": (trainable / total * 100) if total > 0 else 0
        }
