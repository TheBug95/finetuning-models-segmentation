from .common import build_medical_dataset

def build_retinopathy_dataset(root: str):
    """Create a dataset for diabetic retinopathy segmentation.

    Expects the same folder layout as :func:`build_cataract_dataset`.
    """
    return build_medical_dataset(root, "diabetic retinopathy")
