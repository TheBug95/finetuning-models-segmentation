# finetuning-models-segmentation

Repositorio para probar finetuning de modelos de segmentacion usando lora/qlora.

## Uso

El script `finetune.py` permite ajustar modelos tipo SAM2, MedSAM2 y MobileSAM
sobre los datasets de catarata y retinopatía diabética.

Los datasets pueden estar organizados de dos maneras:

1. Directorios separados de imágenes y máscaras binaras:

```
<root>/
    images/
        xxx.png
    masks/
        xxx.png
```

2. Formato COCO (por ejemplo exportado desde Roboflow), donde cada división
contiene todas las imágenes y un archivo `_annotations.coco.json`:

```
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

Ejemplo de ejecución:

```
python finetune.py --model sam2 --method lora --dataset cataract --dataset-root /ruta/al/dataset
```

El parámetro `--method` acepta `baseline`, `lora` y `qlora` para comparar los
métodos de entrenamiento.

Los modelos y el procesador resultante se guardan en un directorio con el
formato `finetuned-<modelo>-<metodo>-<dataset>`.
