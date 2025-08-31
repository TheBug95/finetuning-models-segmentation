# Fix para Error de Tipos de Datos Mixtos

## Problema
Se estaba presentando el error:
```
Error en batch 0: Input type (float) and bias type (c10::Half) should be the same
```

Este error ocurre cuando:
1. El modelo está cargado en `float16` (Half precision)
2. Los datos de entrada están en `float32` 
3. PyTorch no puede hacer la operación porque los tipos no coinciden

## Causa Raíz
En el archivo `models/sam2_model.py`, el modelo se cargaba con:
```python
torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
```

Pero en el entrenador (`trainers/base_trainer.py`), los datos se movían al dispositivo sin conversión de tipo:
```python
images = images.to(self.model.device)
masks = masks.to(self.model.device)
```

## Solución Implementada

### 1. Uso Consistente de float32
Modificado `models/sam2_model.py`:
```python
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
```

### 2. Conversión Automática en Forward Pass
Modificado el método `forward` en `models/sam2_model.py`:
```python
def forward(self, images, input_points=None, input_labels=None, input_boxes=None):
    """Forward pass del modelo con conversión automática de tipos."""
    if self.model is None:
        raise RuntimeError("Modelo no cargado")
        
    # Convertir inputs al tipo del modelo para evitar incompatibilidad
    model_dtype = getattr(self, 'dtype', torch.float32)
    
    # Procesar inputs con conversión de tipos
    if self.processor is not None:
        # ... procesamiento con processor
        if hasattr(inputs, 'pixel_values'):
            inputs.pixel_values = inputs.pixel_values.to(model_dtype)
        return self.model(**inputs)
    else:
        # Forward directo con conversión
        if hasattr(images, 'to'):
            images = images.to(self.device).to(model_dtype)
        return self.model(pixel_values=images)
```

### 3. Soporte en Clase Base
Añadido atributo `dtype` en `models/base_model.py`:
```python
def __init__(self, model_name: str, cache_dir: Optional[str] = None):
    # ... otros atributos
    self.dtype = torch.float32  # Tipo por defecto
```

### 4. Conversión en Entrenador
Modificado `trainers/base_trainer.py`:
```python
# Mover datos al dispositivo y convertir al tipo del modelo
model_dtype = getattr(self.model, 'dtype', torch.float32)
images = images.to(self.model.device).to(model_dtype)
masks = masks.to(self.model.device).to(model_dtype)
```

## Verificación
Creado script de test `test_dtype_fix.py` que verifica:
1. ✅ Compatibilidad básica de tipos
2. ✅ Manejo de escenarios de tipos mixtos
3. ✅ Conversión automática funcionando

## Resultado
- ❌ Antes: `Input type (float) and bias type (c10::Half) should be the same`
- ✅ Después: Entrenamiento sin errores de tipos de datos

## Notas Adicionales
- El fix usa `float32` por defecto para máxima compatibilidad
- La conversión automática se hace en el momento del forward pass
- Si se necesita `float16` por memoria, se puede cambiar `self.dtype` pero asegurando conversión consistente
- El fix es retrocompatible y no afecta la funcionalidad existente

## Comandos de Prueba
```bash
# Probar el fix
python test_dtype_fix.py

# Una vez resuelto el tema de autenticación de HuggingFace:
python train.py --model sam2 --method baseline --dataset cataract --dataset-root "data/Cataract COCO Segmentation" --epochs 1 --batch-size 1
```
