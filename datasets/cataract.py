import os
from .common import SegmentationDataset

def build_cataract_dataset(root: str):
    """Create a dataset for cataract segmentation.

    Expects the following directory structure inside ``root``::

        root/
            images/
                xxx.png
            masks/
                xxx.png
    """
    image_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    return SegmentationDataset(image_dir, mask_dir)
