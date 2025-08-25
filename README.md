# finetuning-models-segmentation

Repositorio para fine-tuning de modelos de segmentación médica usando LoRA/QLoRA.

## Modelos Soportados

- **SAM2**: `facebook/sam2-hiera-tiny`
- **MedSAM**: `flaviagiammarino/medsam-vit-base`
- **MobileSAM**: `dhkim2810/MobileSAM`

## Datasets Soportados

- **Cataract**: Segmentación de cataratas
- **Retinopathy**: Segmentación de retinopatía diabética

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
# Entrenamiento básico
python finetune.py --model sam2 --method baseline --dataset cataract --dataset-root /ruta/al/dataset

# Con LoRA
python finetune.py --model medsam --method lora --dataset retinopathy --dataset-root /ruta/al/dataset --epochs 5 --batch-size 4

# Con QLoRA (para GPUs con memoria limitada)
python finetune.py --model mobilesam --method qlora --dataset cataract --dataset-root /ruta/al/dataset --lr 5e-5
```

### Parámetros Disponibles

- `--model`: Modelo a entrenar (`sam2`, `medsam`, `mobilesam`)
- `--method`: Método de entrenamiento (`baseline`, `lora`, `qlora`)
- `--dataset`: Dataset a usar (`cataract`, `retinopathy`)
- `--dataset-root`: Directorio raíz del dataset
- `--epochs`: Número de épocas (default: 1)
- `--batch-size`: Tamaño del batch (default: 2)
- `--lr`: Learning rate (default: 1e-4)
- `--output-dir`: Directorio de salida (se genera automáticamente si no se especifica)

El parámetro `--method` acepta:

- `baseline`: Entrenamiento completo sin optimizaciones
- `lora`: Low-Rank Adaptation para fine-tuning eficiente
- `qlora`: LoRA cuantizado para GPUs con memoria limitada

Los modelos y el procesador resultante se guardan en un directorio con el
formato `finetuned-<modelo>-<metodo>-<dataset>`.

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

- ✅ Manejo robusto de errores
- ✅ Validación de argumentos de entrada
- ✅ Logging detallado durante el entrenamiento
- ✅ Verificación automática de estructura de datasets
- ✅ Eliminación de código duplicado
- ✅ Documentación mejorada
- ✅ Dependencias específicas con versiones
