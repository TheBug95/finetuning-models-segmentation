# Fix para Error de Attention Mask Duplicada

## Problema
Se estaba presentando el error:
```
❌ Error en batch 1: transformers.integrations.sdpa_attention.sdpa_attention_forward() got multiple values for keyword argument 'attention_mask'
```

## Causa Raíz
El procesador de transformers estaba generando argumentos como `attention_mask` y `position_ids` que entraban en conflicto con argumentos internos del modelo SAM2, causando argumentos duplicados en la función `sdpa_attention_forward`.

## Solución Implementada

### 1. Filtrado de Argumentos Problemáticos
Modificado los métodos `forward` en todos los modelos para filtrar argumentos que causan conflictos:

```python
# Filtrar argumentos problemáticos que causan conflictos
# con la función interna sdpa_attention_forward
filtered_inputs = {}
for key, value in inputs.items():
    # Mantener solo los argumentos esenciales y seguros
    if key in ['pixel_values', 'original_sizes', 'reshaped_input_sizes']:
        filtered_inputs[key] = value
    # Evitar attention_mask y otros argumentos que pueden causar conflictos
    elif key not in ['attention_mask', 'position_ids']:
        filtered_inputs[key] = value

return self.model(**filtered_inputs)
```

### 2. Archivos Modificados
- `models/sam2_model.py`: Agregado filtrado de argumentos
- `models/mobilesam_model.py`: Agregado filtrado de argumentos
- `models/medsam2_model.py`: Agregado filtrado de argumentos

### 3. Argumentos Filtrados
Los siguientes argumentos se eliminan antes de pasar al modelo:
- `attention_mask`: Causa conflicto directo con función interna
- `position_ids`: Puede causar conflictos similares

### 4. Argumentos Conservados
Se mantienen los argumentos esenciales:
- `pixel_values`: Datos de imagen (esencial)
- `original_sizes`: Tamaños originales de imagen
- `reshaped_input_sizes`: Tamaños redimensionados
- Cualquier otro argumento que NO esté en la lista de conflictivos

## Verificación
Creado script de test `test_attention_fix.py` que verifica:
1. ✅ Forward pass funciona sin errores de argumentos duplicados
2. ✅ Filtrado correcto de argumentos problemáticos
3. ✅ Conservación de argumentos esenciales

## Resultado
- ❌ Antes: `got multiple values for keyword argument 'attention_mask'`
- ✅ Después: Forward pass exitoso sin conflictos de argumentos

## Comandos de Prueba
```bash
# Probar el fix específico
python test_attention_fix.py

# Una vez resuelto el tema de autenticación:
python train.py --model sam2 --method baseline --dataset cataract --dataset-root "data/Cataract COCO Segmentation" --epochs 1 --batch-size 1
```

## Notas Técnicas
- El fix es compatible con todos los modelos SAM (SAM2, MedSAM2, MobileSAM)
- No afecta la funcionalidad del modelo, solo filtra argumentos problemáticos
- Es retrocompatible y seguro
- El filtrado es conservador: mantiene argumentos desconocidos a menos que estén específicamente en la lista de conflictivos
