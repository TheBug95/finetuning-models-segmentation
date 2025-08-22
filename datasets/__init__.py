from .cataract import build_cataract_dataset
from .retinopathy import build_retinopathy_dataset


def get_dataset(name: str, root: str):
    """Factory function for available datasets."""
    name = name.lower()
    if name == "cataract":
        return build_cataract_dataset(root)
    if name in {"retinopathy", "diabetic_retinopathy"}:
        return build_retinopathy_dataset(root)
    raise ValueError(f"Unknown dataset: {name}")
