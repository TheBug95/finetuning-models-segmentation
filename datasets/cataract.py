from .common import build_medical_dataset

def build_cataract_dataset(root: str):
    """Create a dataset for cataract segmentation.

    Expects the following directory structure inside ``root``::

        root
            images/
                xxx.png
            masks/
                xxx.png
        
        OR COCO format:
        
        root/
            _annotations.coco.json
            image1.jpg
            image2.jpg
            ...
    """
    return build_medical_dataset(root, "cataract")
