"""Generic dataset capable of loading either COCO annotations or
paired image/mask files.

This class contains the shared logic previously duplicated in
``cataract_dataset.py`` and ``retinopathy_dataset.py``.  It can be
configured via the ``disease_type`` argument so small wrapper classes
can be used for specific datasets.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from PIL import Image

from .base_dataset import BaseMedicalDataset, COCO_AVAILABLE


class GenericCocoOrMaskDataset(BaseMedicalDataset):
    """Dataset that supports COCO annotations or image/mask pairs.

    Parameters
    ----------
    root: str
        Root directory of the dataset.
    split: str
        Dataset split to use (``"train"``, ``"val"`` or ``"test"``).
    disease_type: str, optional
        Name of the disease.  Used only for informational purposes in
        :meth:`get_class_distribution`.
    **kwargs: Any
        Additional arguments forwarded to :class:`BaseMedicalDataset`.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        *,
        disease_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.coco_mode = False
        self.disease_type = disease_type
        super().__init__(root, split, **kwargs)

    # ------------------------------------------------------------------
    # Data loading helpers
    def _load_data_list(self) -> List[Dict[str, Any]]:  # pragma: no cover - IO heavy
        """Load dataset entries either from COCO annotations or from
        image/mask directories."""

        coco_file = os.path.join(self.root, "_annotations.coco.json")
        if os.path.exists(coco_file) and COCO_AVAILABLE:
            return self._load_coco_data(coco_file)
        return self._load_standard_data()

    def _load_coco_data(self, coco_file: str) -> List[Dict[str, Any]]:  # pragma: no cover - IO heavy
        """Load data using a COCO annotations file."""
        try:
            from pycocotools.coco import COCO

            coco = COCO(coco_file)
            self.coco_mode = True

            data_list: List[Dict[str, Any]] = []
            for img_id in coco.getImgIds():
                img_info = coco.loadImgs(img_id)[0]
                image_path = os.path.join(self.root, img_info["file_name"])
                if not os.path.exists(image_path):
                    continue

                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                if anns:
                    data_list.append(
                        {
                            "image_path": image_path,
                            "mask_path": None,
                            "annotations": anns,
                            "image_info": img_info,
                            "coco": coco,
                        }
                    )

            print(f"✅ Datos COCO cargados: {len(data_list)} imágenes con anotaciones")
            return data_list
        except Exception as e:  # pragma: no cover - optional dependency
            print(f"⚠️  Error cargando COCO: {e}. Usando formato estándar.")
            return self._load_standard_data()

    def _load_standard_data(self) -> List[Dict[str, Any]]:  # pragma: no cover - IO heavy
        """Load paired image/mask data."""

        split_dir = os.path.join(self.root, self.split)
        if os.path.exists(split_dir):
            data_list = self._find_images_and_masks(split_dir)
        else:
            data_list = self._find_images_and_masks(self.root)

        formatted: List[Dict[str, Any]] = []
        for item in data_list:
            formatted.append(
                {
                    "image_path": item["image_path"],
                    "mask_path": item["mask_path"],
                    "annotations": None,
                    "image_info": None,
                    "coco": None,
                }
            )

        print(f"✅ Datos estándar cargados: {len(formatted)} pares imagen-máscara")
        return formatted

    # ------------------------------------------------------------------
    # Image/Mask loading
    def _load_image(self, image_path: str) -> Image.Image:  # pragma: no cover - IO heavy
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:  # pragma: no cover - error path
            raise RuntimeError(f"Error cargando imagen {image_path}: {e}")

    def _load_mask(
        self,
        mask_path: Optional[str] = None,
        data_item: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:  # pragma: no cover - IO heavy
        if self.coco_mode and data_item is not None:
            return self._generate_coco_mask(data_item)
        if mask_path and os.path.exists(mask_path):
            try:
                return Image.open(mask_path).convert("L")
            except Exception as e:  # pragma: no cover - error path
                raise RuntimeError(f"Error cargando máscara {mask_path}: {e}")
        raise RuntimeError("No se puede cargar la máscara")

    def _generate_coco_mask(self, data_item: Dict[str, Any]) -> Image.Image:  # pragma: no cover - IO heavy
        try:
            import numpy as np

            coco = data_item["coco"]
            anns = data_item["annotations"]
            img_info = data_item["image_info"]

            combined = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
            for ann in anns:
                if "segmentation" in ann:
                    mask = coco.annToMask(ann)
                    combined = np.maximum(combined, mask * 255)
            return Image.fromarray(combined, mode="L")
        except Exception as e:  # pragma: no cover - error path
            raise RuntimeError(f"Error generando máscara COCO: {e}")

    # ------------------------------------------------------------------
    # Dataset protocol
    def __getitem__(self, idx: int):  # pragma: no cover - IO heavy
        data_item = self.data_list[idx]
        image = self._load_image(data_item["image_path"])
        if self.coco_mode:
            mask = self._load_mask(data_item=data_item)
        else:
            mask = self._load_mask(data_item["mask_path"])

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask

    def get_class_distribution(self) -> Dict[str, Any]:
        distribution: Dict[str, Any] = {
            "total_samples": len(self.data_list),
            "format": "COCO" if self.coco_mode else "Standard",
            "split": self.split,
        }
        if self.disease_type:
            distribution["disease_type"] = self.disease_type
        return distribution

