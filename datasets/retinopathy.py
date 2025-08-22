import os
from .common import SegmentationDataset, CocoSegmentationDataset

def build_retinopathy_dataset(root: str):
    """Create a dataset for diabetic retinopathy segmentation.

    Expects the same folder layout as :func:`build_cataract_dataset`.
    """
    ann_file = os.path.join(root, "_annotations.coco.json")
    if os.path.exists(ann_file):
        return CocoSegmentationDataset(root)
    image_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    return SegmentationDataset(image_dir, mask_dir)
