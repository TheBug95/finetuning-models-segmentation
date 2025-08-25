import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

try:
    from pycocotools.coco import COCO
except Exception:  # pragma: no cover - optional dependency
    COCO = None


def build_medical_dataset(root: str, dataset_name: str = "medical"):
    """Create a generic medical segmentation dataset.
    
    This function can be used for any medical segmentation dataset that follows
    either the COCO format or the image/mask directory structure.

    Parameters
    ----------
    root : str
        Root directory of the dataset
    dataset_name : str
        Name of the dataset (for logging purposes)

    Returns
    -------
    Dataset
        Either CocoSegmentationDataset or SegmentationDataset

    Raises
    ------
    ValueError
        If neither COCO format nor image/mask directories are found
    """
    ann_file = os.path.join(root, "_annotations.coco.json")
    if os.path.exists(ann_file):
        print(f"Using COCO format {dataset_name} dataset from {root}")
        return CocoSegmentationDataset(root)
    
    image_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        raise ValueError(
            f"{dataset_name.capitalize()} dataset structure not found. Expected either:\n"
            f"  - COCO format: {ann_file}\n"
            f"  - Image/mask dirs: {image_dir} and {mask_dir}"
        )
    
    print(f"Using image/mask format {dataset_name} dataset from {root}")
    return SegmentationDataset(image_dir, mask_dir)


class SegmentationDataset(Dataset):
    """Generic image segmentation dataset.

    Expects two directories: one with images and another with masks.
    Filenames must match between images and masks.
    """

    def __init__(self, image_dir: str, mask_dir: str):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        # Get all image files
        image_files = [
            f for f in os.listdir(image_dir) 
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))
        ]
        
        # Verify corresponding masks exist
        self.images = []
        for img_file in sorted(image_files):
            mask_path = os.path.join(mask_dir, img_file)
            if os.path.exists(mask_path):
                self.images.append(img_file)
            else:
                print(f"Warning: No mask found for {img_file}")
        
        if not self.images:
            raise ValueError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")
        
        self.to_tensor = T.ToTensor()
        print(f"Found {len(self.images)} valid image-mask pairs")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_name = self.images[idx]
        img_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)
            
            return self.to_tensor(image), self.to_tensor(mask)
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path} or mask {mask_path}: {e}")


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
        
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")
        
        try:
            self.coco = COCO(ann_path)
        except Exception as e:
            raise RuntimeError(f"Error loading COCO annotations from {ann_path}: {e}")
        
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        self.to_tensor = T.ToTensor()
        print(f"Loaded COCO dataset with {len(self.img_ids)} images")

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        image_path = os.path.join(self.root, img_info["file_name"])
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        
        for ann in anns:
            ann_mask = self.coco.annToMask(ann)
            mask = np.maximum(mask, ann_mask)
        
        mask_img = Image.fromarray(mask * 255)
        return self.to_tensor(image), self.to_tensor(mask_img)
