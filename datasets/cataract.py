import os
from .common import SegmentationDataset, CocoSegmentationDataset

def build_cataract_dataset(root: str):
    """Create a dataset for cataract segmentation.

    Expects the following directory structure inside ``root``::

        root/
            images/
                xxx.png
            masks/
                xxx.png
    """
    ann_file = os.path.join(root, "_annotations.coco.json")
    if os.path.exists(ann_file):
        return CocoSegmentationDataset(root)
    image_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    return SegmentationDataset(image_dir, mask_dir)
