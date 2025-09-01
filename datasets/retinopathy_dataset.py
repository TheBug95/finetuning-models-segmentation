"""Retinopathy dataset built on top of :class:`GenericCocoOrMaskDataset`."""

from .generic_dataset import GenericCocoOrMaskDataset


class RetinopathyDataset(GenericCocoOrMaskDataset):
    """Thin wrapper for diabetic retinopathy segmentation."""

    def __init__(self, root: str, split: str = "train", **kwargs):
        super().__init__(
            root=root,
            split=split,
            disease_type="diabetic_retinopathy",
            **kwargs,
        )

