import os

from PIL import Image
from PIL.ImageFile import ImageFile
from torch.utils.data import Dataset


class SRDataset(Dataset):
    def __init__(self, lr_dir: str, hr_dir: str, transform=None):
        self.lr_dir: str = lr_dir
        self.hr_dir: str = hr_dir
        self.lr_paths: list[str] = sorted(os.listdir(lr_dir))
        self.hr_paths: list[str]= sorted(os.listdir(hr_dir))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.lr_paths)

    def __getitem__(self, idx) -> tuple[ImageFile, ImageFile]:
        lr_img_path = os.path.join(self.lr_dir, self.lr_paths[idx])
        hr_img_path = os.path.join(self.hr_dir, self.hr_paths[idx])

        lr_image: ImageFile = Image.open(lr_img_path).convert('RGB')
        hr_image: ImageFile = Image.open(hr_img_path).convert('RGB')

        if self.transform is not None:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image
