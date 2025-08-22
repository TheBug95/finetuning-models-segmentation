import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class SegmentationDataset(Dataset):
    """Generic image segmentation dataset.

    Expects two directories: one with images and another with masks.
    Filenames must match between images and masks.
    """

    def __init__(self, image_dir: str, mask_dir: str):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(
            [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))]
        )
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_name = self.images[idx]
        img_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        return self.to_tensor(image), self.to_tensor(mask)
