# Fine-tuning de Modelos SAM para Segmentación Médica

Sistema modular y eficiente para fine-tuning de modelos SAM (Segment Anything Model) especializados en segmentación médica usando **Hugging Face Transformers**.

## 🚀 Características Principales

- **✅ Modular y POO**: Arquitectura completamente orientada a objetos
- **✅ Hugging Face First**: Usa modelos oficiales de HF cuando están disponibles
- **✅ Métodos Múltiples**: Soporte para Baseline, LoRA y QLoRA
- **✅ Datasets Flexibles**: Soporte COCO y formatos estándar
- **✅ Benchmark Automatizado**: Comparación sistemática de modelos y métodos

## 🤖 Modelos Soportados

### SAM2 (facebook/sam2-*)
- **tiny**: `facebook/sam2-hiera-tiny` (más rápido)
- **base**: `facebook/sam2-hiera-base-plus` 
- **large**: `facebook/sam2-hiera-large`
- **huge**: `facebook/sam2.1-hiera-large` (mejor calidad)

### MedSAM2 (wanglab/*)
- **default**: `wanglab/MedSAM2` (especializado medicina)
- **vit_base**: `wanglab/medsam-vit-base`
- **medsam_mix**: `guinansu/MedSAMix`

### MobileSAM (nielsr/*, qualcomm/*)
- **default**: `nielsr/mobilesam` (optimizado móvil)
- **qualcomm**: `qualcomm/MobileSam`
- **v2**: `RogerQi/MobileSAMV2`

## ⚙️ Métodos de Fine-tuning

| Método | Descripción | Parámetros Entrenables | Memoria GPU | Velocidad |
|--------|-------------|----------------------|-------------|-----------|
| **Baseline** | Fine-tuning completo | 100% | Alta | Lento |
| **LoRA** | Low-Rank Adaptation | ~1-5% | Media | Medio |
| **QLoRA** | LoRA + Cuantización 4-bit | ~1-5% | Baja | Rápido |

## 📊 Datasets Soportados

- **Cataract**: Segmentación de cataratas (COCO/estándar)
- **Retinopathy**: Retinopatía diabética (COCO/estándar)

## 🔧 Instalación

```bash
# Clonar repositorio
git clone <repo_url>
cd finetuning-models-segmentation

# Instalar dependencias
pip install -r requirements.txt
```

## 🚀 Uso Rápido

### Entrenamiento Individual

```bash
# Entrenar SAM2 con LoRA en dataset de cataratas
python train.py \
    --model sam2 \
    --variant tiny \
    --method lora \
    --dataset cataract \
    --dataset-root ./data/Cataract\ COCO\ Segmentation/train \
    --epochs 5 \
    --batch-size 4

# Entrenar MedSAM2 con QLoRA en retinopatía
python train.py \
    --model medsam2 \
    --method qlora \
    --dataset retinopathy \
    --dataset-root ./data/Diabetic-Retinopathy\ COCO\ Segmentation/train \
    --epochs 10 \
    --learning-rate 2e-4
```

### Benchmark Completo

```bash
# Comparar todos los modelos y métodos
python benchmark.py \
    --dataset-root ./data \
    --epochs 3 \
    --output-dir benchmark_results

# Comparar solo LoRA vs QLoRA
python benchmark.py \
    --dataset-root ./data \
    --methods lora qlora \
    --epochs 5
```

## 📁 Estructura del Proyecto

```
finetuning-models-segmentation/
├── models/                     # Modelos modulares
│   ├── __init__.py
│   ├── base_model.py          # Clase base
│   ├── sam2_model.py          # SAM2 implementation
│   ├── medsam2_model.py       # MedSAM2 implementation
│   └── mobilesam_model.py     # MobileSAM implementation
├── trainers/                   # Entrenadores modulares
│   ├── __init__.py
│   ├── base_trainer.py        # Trainer base
│   ├── baseline_trainer.py    # Baseline training
│   ├── lora_trainer.py        # LoRA training
│   └── qlora_trainer.py       # QLoRA training
├── datasets/                   # Datasets modulares
│   ├── __init__.py
│   ├── base_dataset.py        # Dataset base
│   ├── cataract_dataset.py    # Cataract dataset
│   ├── retinopathy_dataset.py # Retinopathy dataset
│   └── dataset_factory.py     # Factory pattern
├── train.py                    # Script principal
├── benchmark.py               # Benchmark automatizado
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

## 🎯 Arquitectura Modular

### Modelos
```python
from models import SAM2Model, MedSAM2Model, MobileSAMModel

# Cargar modelo
model = SAM2Model(variant="tiny")
model.load_model()
model.load_processor()

# Configurar para fine-tuning
model.setup_for_finetuning("lora", r=16, lora_alpha=32)
```

### Entrenadores
```python
from trainers import LoRATrainer

# Crear entrenador
trainer = LoRATrainer(model, learning_rate=1e-4)
trainer.setup_model_for_training()
trainer.setup_optimizer()

# Entrenar
metrics = trainer.train(dataloader, epochs=5)
```

### Datasets
```python
from datasets import create_dataset

# Crear dataset
dataset = create_dataset("cataract", "./data/cataract", split="train")
dataloader = DataLoader(dataset, batch_size=4)
```

## 📊 Resultados de Benchmark

El script `benchmark.py` genera automáticamente:

- **CSV Report**: `benchmark_report.csv` con métricas detalladas
- **JSON Stats**: `benchmark_stats.json` con estadísticas agregadas
- **Raw Results**: `benchmark_results.json` con resultados completos

### Métricas Incluidas
- Tiempo de entrenamiento
- Loss final y mejor loss
- Parámetros entrenables vs totales
- Tasa de éxito por modelo/método
- Uso de memoria (estimado)

## 🔍 Comandos Útiles

```bash
# Listar modelos disponibles
python train.py --list-models

# Listar datasets disponibles
python train.py --list-datasets

# Entrenar con configuración personalizada de LoRA
python train.py \
    --model sam2 \
    --method lora \
    --dataset cataract \
    --dataset-root ./data/cataract \
    --lora-r 32 \
    --lora-alpha 64 \
    --lora-dropout 0.05

# Benchmark solo modelos específicos
python benchmark.py \
    --dataset-root ./data \
    --models sam2 medsam2 \
    --methods lora qlora
```

## 🎨 Extensibilidad

### Añadir Nuevo Modelo
```python
from models.base_model import BaseSegmentationModel

class MyNewModel(BaseSegmentationModel):
    def load_model(self):
        # Implementar carga del modelo
        pass
    
    def load_processor(self):
        # Implementar carga del procesador
        pass
```

### Añadir Nuevo Dataset
```python
from datasets.base_dataset import BaseMedicalDataset

class MyNewDataset(BaseMedicalDataset):
    def _load_data_list(self):
        # Implementar carga de datos
        pass
```

## 🚨 Mejores Prácticas

1. **Usar HF Models**: Siempre preferir modelos de Hugging Face
2. **Modularidad**: Mantener componentes separados y reutilizables
3. **Error Handling**: Manejo robusto de errores en cada componente
4. **Logging**: Logging detallado para debugging
5. **Testing**: Probar componentes individualmente antes de integrar

## 🔧 Configuración de GPU

```bash
# Verificar GPU disponible
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Para múltiples GPUs (futuro)
export CUDA_VISIBLE_DEVICES=0,1
```

## 📈 Roadmap

- [ ] Soporte para múltiples GPUs
- [ ] Integración con Weights & Biases
- [ ] Más métricas de evaluación (IoU, Dice, etc.)
- [ ] Soporte para más datasets médicos
- [ ] Optimización automática de hiperparámetros
- [ ] Exportación a formatos de inferencia (ONNX, TensorRT)

## 🤝 Contribuciones

1. Fork el repositorio
2. Crear branch para nueva feature (`git checkout -b feature/nueva-feature`)
3. Commit cambios (`git commit -am 'Añadir nueva feature'`)
4. Push al branch (`git push origin feature/nueva-feature`)
5. Crear Pull Request

## 📝 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

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
