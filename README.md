# Fine-tuning SAM2, MedSAM2 y MobileSAM2 para Segmentación Médica

Repositorio especializado para fine-tuning de los modelos SAM de segunda generación en datasets médicos usando técnicas baseline, LoRA y QLoRA para comparación de rendimiento.

## 🤖 Modelos Soportados

- **sam2**: SAM2 - Next generation Segment Anything Model
- **medsam2**: MedSAM2 - Medical specialized SAM2  
- **mobilesam2**: MobileSAM2 - Lightweight SAM2 for mobile devices

> **🔗 Integración Automática**: Todos los modelos se descargan e instalan automáticamente usando el `SAMModelManager` integrado.

## 📊 Datasets Soportados

- **Cataract**: Segmentación de cataratas
- **Retinopathy**: Segmentación de retinopatía diabética

## ⚙️ Métodos de Fine-tuning

### 🏗️ Baseline (Recomendado para comparación)
- **Descripción**: Fine-tuning completo de todas las capas del modelo
- **Ventajas**: Máxima flexibilidad y potencial rendimiento
- **Desventajas**: Mayor uso de memoria y tiempo de entrenamiento
- **Uso**: Ideal como línea base para comparar con técnicas eficientes

### 🎯 LoRA (Low-Rank Adaptation)
- **Descripción**: Fine-tuning eficiente mediante adaptadores de bajo rango
- **Ventajas**: Balance entre eficiencia y rendimiento
- **Desventajas**: Ligeramente menos flexible que baseline
- **Uso**: Recomendado para la mayoría de casos prácticos

### 💾 QLoRA (Quantized LoRA)
- **Descripción**: LoRA con cuantización 4-bit para mínimo uso de memoria
- **Ventajas**: Mínimo uso de memoria GPU
- **Desventajas**: Posible pequeña pérdida de rendimiento
- **Uso**: Ideal para GPUs con memoria limitada

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

El script `finetune.py` permite ajustar modelos tipo SAM2, MedSAM y MobileSAM
sobre los datasets de catarata y retinopatía diabética.

### Estructura de Datasets

Los datasets pueden estar organizados de dos maneras:

1. **Directorios separados** de imágenes y máscaras binarias:

```text
<root>/
    images/
        xxx.png
    masks/
        xxx.png
```

2. **Formato COCO** (por ejemplo exportado desde Roboflow), donde cada división
contiene todas las imágenes y un archivo `_annotations.coco.json`:

```text
<root>/
    train/
        _annotations.coco.json
        img1.png
        ...
    valid/
        _annotations.coco.json
        ...
```

En este caso, proporcione la ruta del subconjunto deseado (`train`, `valid`,
etc.) mediante el parámetro `--dataset-root`.

### Ejemplo de Ejecución

```bash
# Fine-tuning baseline (para establecer línea base de comparación)
python finetune.py --model sam2 --method baseline --dataset cataract --dataset-root /ruta/al/dataset --epochs 5

# Fine-tuning con LoRA (recomendado para la mayoría de casos)
python finetune.py --model medsam2 --method lora --dataset retinopathy --dataset-root /ruta/al/dataset --epochs 5 --batch-size 4

# Fine-tuning con QLoRA (para GPUs con memoria limitada)
python finetune.py --model mobilesam2 --method qlora --dataset cataract --dataset-root /ruta/al/dataset --lr 5e-5

# Comparación automática de todos los modelos y métodos
python compare_models.py --dataset-root /ruta/al/dataset --epochs 3

# Listar modelos disponibles
python finetune.py --list-models

# Ver estado del gestor de modelos
python finetune.py --show-manager-status
```

### Parámetros Disponibles

- `--model`: Modelo a entrenar (`sam2`, `medsam2`, `mobilesam2`)
- `--method`: Método de entrenamiento (`baseline`, `lora`, `qlora`)
- `--dataset`: Dataset a usar (`cataract`, `retinopathy`)
- `--dataset-root`: Directorio raíz del dataset
- `--epochs`: Número de épocas (default: 1)
- `--batch-size`: Tamaño del batch (default: 2)
- `--lr`: Learning rate (default: 1e-4)
- `--output-dir`: Directorio de salida (se genera automáticamente si no se especifica)
- `--models-dir`: Directorio para modelos descargados (default: "models")
- `--list-models`: Listar modelos disponibles
- `--show-manager-status`: Mostrar estado del SAMModelManager

El parámetro `--method` acepta:

- `baseline`: Entrenamiento completo sin optimizaciones
- `lora`: Low-Rank Adaptation para fine-tuning eficiente
- `qlora`: LoRA cuantizado para GPUs con memoria limitada

Los modelos y el procesador resultante se guardan en un directorio con el
formato `finetuned-<modelo>-<metodo>-<dataset>-epochs<num_epochs>`.

## 🔬 Comparación de Métodos

### Script de Comparación Automática

El proyecto incluye `compare_models.py` para ejecutar comparaciones sistemáticas:

```bash
# Comparar todos los modelos con todos los métodos
python compare_models.py --dataset-root /ruta/al/dataset

# Comparar solo métodos específicos
python compare_models.py --dataset-root /ruta/al/dataset --methods baseline lora

# Comparar solo modelos específicos
python compare_models.py --dataset-root /ruta/al/dataset --models sam2 medsam2

# Vista previa de experimentos sin ejecutar
python compare_models.py --dataset-root /ruta/al/dataset --dry-run
```

### Métricas de Comparación

Cada experimento genera automáticamente:

- **training_metrics.json**: Métricas detalladas del entrenamiento
- **comparison_report.md**: Reporte comparativo en Markdown
- **Tiempo de entrenamiento**: Para analizar eficiencia
- **Pérdida final**: Para comparar convergencia
- **Uso de memoria**: Para optimizar recursos

### Resultados Esperados

| Método | Memoria GPU | Tiempo | Calidad | Recomendado para |
|--------|-------------|--------|---------|------------------|
| **Baseline** | Alta | Alto | Máxima | Comparación y máximo rendimiento |
| **LoRA** | Media | Medio | Alta | Uso general y producción |
| **QLoRA** | Baja | Medio | Buena | GPUs con memoria limitada |

## Gestión de Modelos

Para descargar e instalar los pesos oficiales de las distintas variantes de
SAM, MobileSAM, HQ-SAM y MedSAM(+2) se incluye el módulo
`model_manager.py`. Este expone la clase `SAMModelManager`, que automatiza la
instalación de los repositorios y la obtención de los checkpoints.

### Ejemplo de Uso del Model Manager

```python
from model_manager import SAMModelManager

# Crear el gestor
mgr = SAMModelManager("Models")

# Listar modelos soportados
mgr.list_supported()

# Descargar e instalar un modelo específico
ckpt_path = mgr.setup("mobilesam", "vit_t")

# Solo descargar sin instalar dependencias
ckpt_path = mgr.download_variant("sam", "vit_b")
```

Los archivos descargados se guardarán dentro del directorio indicado (en el
ejemplo, `Models`).

### Modelos Disponibles en Model Manager

- **sam**: ViT-B, ViT-L, ViT-H
- **mobilesam**: ViT-T
- **hq_sam**: ViT-B, ViT-L, ViT-H, ViT-T, Hiera-L-2.1
- **medsam**: MedSAM1, MedSAM2-latest

## Mejoras Implementadas

- ✅ **Integración completa SAMModelManager**: Setup automático de modelos
- ✅ **Manejo robusto de errores**: Validación en todas las funciones críticas
- ✅ **Logging detallado**: Información completa durante el entrenamiento
- ✅ **Verificación automática**: Detección de estructura de datasets
- ✅ **Eliminación de código duplicado**: Arquitectura modular y limpia
- ✅ **Documentación completa**: Guías detalladas y ejemplos
- ✅ **Dependencias específicas**: Versiones compatibles y tested
- ✅ **Cache inteligente**: Los modelos se descargan una sola vez
- ✅ **Fallbacks automáticos**: Robustez ante fallos de descarga

## 📋 Requisitos

Ver `requirements.txt` para la lista completa de dependencias.

## 🎯 Mejores Prácticas

### Configuración de GPU

- **Memoria recomendada**: 12GB+ para baseline, 8GB+ para LoRA, 6GB+ para QLoRA
- **Uso de mixed precision**: Habilitado automáticamente para optimizar memoria
- **Gradient checkpointing**: Activado para reducir uso de memoria

### Hiperparámetros

- **Learning rate baseline**: 1e-5 (más bajo que LoRA/QLoRA)
- **Gradient clipping**: 1.0 para estabilizar entrenamiento baseline
- **Batch size**: Ajustar según memoria disponible
- **Épocas**: 3-5 para datasets pequeños, 10+ para datasets grandes

### Monitoreo del Entrenamiento

- Usar `wandb` o `tensorboard` para seguimiento de métricas
- Revisar `training_metrics.json` para análisis detallado
- Comparar curvas de pérdida entre métodos

### Selección de Método

1. **Baseline**: Para obtener el máximo rendimiento posible y establecer una línea base
2. **LoRA**: Para la mayoría de casos de uso prácticos con buen balance rendimiento/eficiencia
3. **QLoRA**: Para GPUs con memoria limitada manteniendo buena calidad

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature
3. Documenta los cambios
4. Envía un pull request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo LICENSE para más detalles.
