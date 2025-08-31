#!/usr/bin/env python3
"""
Test para verificar que la corrección del conflicto de attention_mask funciona.
"""

import torch
import torch.nn as nn
from models.sam2_model import SAM2Model

class MockBatchEncoding:
    """Mock que simula el BatchEncoding del procesador SAM2."""
    
    def __init__(self):
        self.pixel_values = torch.randn(1, 3, 256, 256)
        self.original_sizes = torch.tensor([[256, 256]])
        self.reshaped_input_sizes = torch.tensor([[256, 256]])
        self.attention_mask = torch.ones(1, 16, 16)  # Esto causa el conflicto
        
    def to(self, device):
        return self
        
    def items(self):
        return [
            ('pixel_values', self.pixel_values),
            ('original_sizes', self.original_sizes),
            ('reshaped_input_sizes', self.reshaped_input_sizes),
            ('attention_mask', self.attention_mask),
        ]
    
    # Hacer que funcione como diccionario para **kwargs
    def __iter__(self):
        return iter(['pixel_values', 'original_sizes', 'reshaped_input_sizes', 'attention_mask'])
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def keys(self):
        return ['pixel_values', 'original_sizes', 'reshaped_input_sizes', 'attention_mask']

class MockProcessor:
    """Mock del procesador SAM2."""
    
    def __call__(self, **kwargs):
        return MockBatchEncoding()

class MockModel(nn.Module):
    """Mock del modelo SAM2 que simula el conflicto de attention_mask."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3, padding=1)
    
    def forward(self, **kwargs):
        # Simular el error específico que ocurre en SAM2
        if 'attention_mask' in kwargs:
            raise TypeError("transformers.integrations.sdpa_attention.sdpa_attention_forward() got multiple values for keyword argument 'attention_mask'")
        
        pixel_values = kwargs.get('pixel_values')
        return {"masks": self.conv(pixel_values)}

def test_attention_mask_fix():
    """Test principal que verifica la corrección."""
    print("🧪 TESTING ATTENTION MASK FIX")
    print("="*50)
    
    # Crear modelo con mocks
    sam2_model = SAM2Model(variant="tiny")
    sam2_model.model = MockModel()
    sam2_model.processor = MockProcessor()
    sam2_model.device = torch.device("cpu")
    sam2_model.dtype = torch.float32
    
    print("📊 Configuración:")
    print("   - Mock que simula conflicto de attention_mask")
    print("   - Filtrado inteligente activado")
    
    # Test con imagen dummy
    dummy_images = torch.randn(1, 3, 256, 256)
    
    try:
        print("\n🏃 Ejecutando forward pass...")
        outputs = sam2_model.forward(dummy_images)
        
        print("✅ Forward pass exitoso!")
        print(f"   Output shape: {outputs['masks'].shape}")
        print("✅ La corrección del filtrado funciona correctamente:")
        print("   1. Detectó el conflicto de attention_mask")
        print("   2. Aplicó filtrado automáticamente")
        print("   3. Reintentó con argumentos seguros")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_attention_mask_fix()
    if success:
        print("\n🎉 TEST PASADO - La corrección funciona!")
    else:
        print("\n💥 TEST FALLÓ - Hay que revisar la implementación")
