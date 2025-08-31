#!/usr/bin/env python3
"""
Solución alternativa más robusta que usa introspección para manejar argumentos.
"""

import inspect
import torch
import torch.nn as nn
from typing import Dict, Any

class SmartArgumentFilter:
    """Filtro inteligente de argumentos que usa introspección."""
    
    def __init__(self, model):
        self.model = model
        self._analyze_model_signature()
        
    def _analyze_model_signature(self):
        """Analiza la firma del modelo para determinar argumentos válidos."""
        if hasattr(self.model, 'forward'):
            forward_signature = inspect.signature(self.model.forward)
            self.valid_params = set(forward_signature.parameters.keys())
            self.has_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD 
                for p in forward_signature.parameters.values()
            )
            print(f"📊 Modelo acepta parámetros: {self.valid_params}")
            print(f"📊 Tiene **kwargs: {self.has_kwargs}")
        else:
            self.valid_params = set()
            self.has_kwargs = True
            
    def filter_safe_arguments(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filtra argumentos de manera inteligente basado en la firma del modelo.
        """
        if self.has_kwargs:
            # Si el modelo acepta **kwargs, intentar con todos los argumentos
            return dict(inputs.items())
        
        # Si no acepta **kwargs, filtrar solo argumentos válidos
        filtered = {}
        for key, value in inputs.items():
            if key in self.valid_params:
                filtered[key] = value
            else:
                print(f"⚠️  Argumento '{key}' no está en la firma del modelo, omitiendo...")
        return filtered
        
    def safe_forward(self, inputs: Dict[str, Any], fallback_args: list = None):
        """
        Ejecuta forward de manera segura con manejo de errores inteligente.
        """
        fallback_args = fallback_args or ['attention_mask', 'position_ids']
        
        # Intentar 1: Con todos los argumentos
        try:
            filtered_inputs = self.filter_safe_arguments(inputs)
            return self.model(**filtered_inputs)
        except TypeError as e:
            if "multiple values" in str(e):
                # Detectar qué argumento específico causa el problema
                error_arg = None
                for arg in fallback_args:
                    if arg in str(e):
                        error_arg = arg
                        break
                
                if error_arg:
                    print(f"⚠️  Conflicto detectado con '{error_arg}', reintentando sin él...")
                    filtered_inputs = {k: v for k, v in inputs.items() if k != error_arg}
                    return self.model(**self.filter_safe_arguments(filtered_inputs))
                else:
                    # Si no podemos identificar el argumento, filtrar todos los problemáticos
                    print(f"⚠️  Error de argumentos múltiples, filtrando argumentos conocidos...")
                    filtered_inputs = {k: v for k, v in inputs.items() if k not in fallback_args}
                    return self.model(**self.filter_safe_arguments(filtered_inputs))
            else:
                # Re-raise errores que no son de argumentos múltiples
                raise e

def create_robust_forward_method(original_forward):
    """
    Decorator que envuelve el forward method con manejo robusto de argumentos.
    """
    def robust_forward(self, *args, **kwargs):
        # Si no hay procesador, usar el forward original
        if self.processor is None:
            return original_forward(self, *args, **kwargs)
        
        # Obtener inputs del procesador
        try:
            # Aquí iría la lógica específica del procesador
            # Por ahora, asumir que tenemos inputs procesados
            inputs = kwargs.get('inputs', {})
            
            # Crear filtro inteligente si no existe
            if not hasattr(self, '_smart_filter'):
                self._smart_filter = SmartArgumentFilter(self.model)
            
            # Usar forward seguro
            return self._smart_filter.safe_forward(inputs)
            
        except Exception as e:
            print(f"⚠️  Error en forward robusto, fallback a forward original: {e}")
            return original_forward(self, *args, **kwargs)
    
    return robust_forward

# Test de la solución robusta
def test_robust_solution():
    """Test de la solución robusta con introspección."""
    print("🧪 TESTING ROBUST ARGUMENT FILTERING")
    print("="*50)
    
    class TestModel(nn.Module):
        def __init__(self, accept_kwargs=True):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, 3, padding=1)
            self.accept_kwargs = accept_kwargs
            
        def forward(self, pixel_values, **kwargs):
            if not self.accept_kwargs and kwargs:
                # Simular modelo que no acepta argumentos extra
                extra_args = list(kwargs.keys())
                raise TypeError(f"forward() got unexpected keyword arguments: {extra_args}")
            return {"masks": self.conv(pixel_values)}
    
    # Test 1: Modelo que acepta **kwargs
    print("📊 Test 1: Modelo con **kwargs")
    model1 = TestModel(accept_kwargs=True)
    filter1 = SmartArgumentFilter(model1)
    
    inputs = {
        'pixel_values': torch.randn(1, 3, 256, 256),
        'attention_mask': torch.ones(1, 16, 16),
        'position_ids': torch.arange(256).unsqueeze(0)
    }
    
    try:
        result = filter1.safe_forward(inputs)
        print("✅ Forward exitoso con modelo que acepta **kwargs")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Modelo más restrictivo
    print("\n📊 Test 2: Modelo sin **kwargs")
    
    class RestrictiveModel(nn.Module):
        def forward(self, pixel_values):  # Solo acepta pixel_values
            return {"masks": torch.zeros(1, 1, 256, 256)}
    
    model2 = RestrictiveModel()
    filter2 = SmartArgumentFilter(model2)
    
    try:
        result = filter2.safe_forward(inputs)
        print("✅ Forward exitoso con modelo restrictivo")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n✅ Test de solución robusta completado")

if __name__ == "__main__":
    test_robust_solution()
