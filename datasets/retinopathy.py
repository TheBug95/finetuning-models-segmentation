import os
from .common import SegmentationDataset

def build_retinopathy_dataset(root: str):
    """Create a dataset for diabetic retinopathy segmentation.

    Expects the same folder layout as :func:`build_cataract_dataset`.
    """
    image_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    return SegmentationDataset(image_dir, mask_dir)
