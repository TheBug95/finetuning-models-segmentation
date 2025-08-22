# finetuning-models-segmentation

Repositorio para probar finetuning de modelos de segmentacion usando lora/qlora.

## Uso

El script `finetune.py` permite ajustar modelos tipo SAM2, MedSAM2 y MobileSAM
sobre los datasets de catarata y retinopatía diabética.

Los datasets deben tener la siguiente estructura:

```
<root>/
    images/
        xxx.png
    masks/
        xxx.png
```

Ejemplo de ejecución:

```
python finetune.py --model sam2 --method lora --dataset cataract --dataset-root /ruta/al/dataset
```

El parámetro `--method` acepta `baseline`, `lora` y `qlora` para comparar los
métodos de entrenamiento.

Los modelos y el procesador resultante se guardan en un directorio con el
formato `finetuned-<modelo>-<metodo>-<dataset>`.
