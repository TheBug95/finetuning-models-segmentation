"""Cataract dataset built on top of :class:`GenericCocoOrMaskDataset`."""

from .generic_dataset import GenericCocoOrMaskDataset


class CataractDataset(GenericCocoOrMaskDataset):
    """Thin wrapper setting ``disease_type`` to ``"cataract"``."""

    def __init__(self, root: str, split: str = "train", **kwargs):
        super().__init__(root=root, split=split, disease_type="cataract", **kwargs)

