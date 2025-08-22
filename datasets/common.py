import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

try:
    from pycocotools.coco import COCO
except Exception:  # pragma: no cover - optional dependency
    COCO = None

class SegmentationDataset(Dataset):
    """Generic image segmentation dataset.

    Expects two directories: one with images and another with masks.
    Filenames must match between images and masks.
    """

    def __init__(self, image_dir: str, mask_dir: str):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(
            [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))]
        )
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_name = self.images[idx]
        img_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        return self.to_tensor(image), self.to_tensor(mask)


class CocoSegmentationDataset(Dataset):
    """Dataset for images annotated in COCO segmentation format.

    Parameters
    ----------
    root: str
        Directory containing the images and a ``_annotations.coco.json`` file.
    ann_file: str, optional
        Name of the annotations file relative to ``root``.
    """

    def __init__(self, root: str, ann_file: str = "_annotations.coco.json"):
        if COCO is None:
            raise ImportError("pycocotools is required for CocoSegmentationDataset")
        self.root = root
        ann_path = os.path.join(root, ann_file)
        self.coco = COCO(ann_path)
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        image_path = os.path.join(self.root, img_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        for ann in anns:
            ann_mask = self.coco.annToMask(ann)
            mask = np.maximum(mask, ann_mask)
        mask_img = Image.fromarray(mask * 255)
        return self.to_tensor(image), self.to_tensor(mask_img)
