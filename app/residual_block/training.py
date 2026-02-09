import math
import os

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_skimage

from app.config import device, lcolumn, model_dir, lr_dir, hr_dir, lr_tst_dir, hr_tst_dir
from app.models.dataset import SRDataset
from app.noise import NoiseAugmenter
from app.residual_block.test import enhance_image
from app.utils.consolegui import print_center


def psnr(img1, img2) -> float:
    mse = torch.mean((img1 - img2) ** 2)
    max_pixel_value = 1.0
    return 20 * math.log10(max_pixel_value / math.sqrt(mse))


def calculate_ssim(img1, img2) -> float:
    return ssim_skimage(
        img1.float().detach().cpu().numpy(),
        img2.float().detach().cpu().numpy(),
        multichannel=True,
        data_range=1.0,
        win_size=-11,
        full=True
    )


class TrainModel:
    def __init__(self, model, criterion, optimizer, epochs: int = 100, batch_size: int = 3):
        self.model = model
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.epoch: int = 0
        self.avg_psnr: float = 0.0
        self.best_psnr = float('-inf')

    def validate_model(self, dataloader) -> None:
        print_center(f"Validation metrics for epoch {self.epoch}")
        self.model.eval()

        # Сначала получаем ссылку на dataset, который использовался для формирования dataloader
        dataset = dataloader.dataset

        # Теперь можно корректно использовать random_split на dataset
        train_size: int = int(len(dataset) * 0.8)
        val_size: int = len(dataset) - train_size
        _, val_dataset = random_split(dataset, [train_size, val_size])

        # Формируем новый валидный DataLoader
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        total_psnr: float = 0.0
        total_ssim: float = 0.0
        count: int = 0

        with torch.no_grad():
            for lr_imgs, hr_imgs in val_dataloader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                outputs = self.model(lr_imgs)

                # Рассчитываем PSNR
                psnr_val: float = psnr(outputs, hr_imgs)
                total_psnr += psnr_val

                # Рассчитываем SSIM
                ssim_val, _ = calculate_ssim(outputs, hr_imgs)
                total_ssim += ssim_val

                count += 1

        self.avg_psnr = total_psnr / count
        avg_ssim: float = total_ssim / count

        print("-" * lcolumn)
        print(f"Average PSNR: {self.avg_psnr:.4f}")
        print(f"Best PSNR: {self.best_psnr:.4f}")  # Предполагается, что self.best_psnr инициализирован ранее
        print(f"Average SSIM: {avg_ssim:.4f}")
        print("-" * lcolumn)

        torch.save(self.model.state_dict(), f'{model_dir}/trained_upscale_model_{self.best_psnr:.4f}.pth')
        self.model.train()

    def train_model(self):
        self.model.train()
        running_loss: float = 0
        best_psnr = float('-inf')
        prob: float = 0.5
        dataloader = None
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        noise_types: list[str] = [None, 'pixelated', 'gaus', 'quantize', 'salt_paper', 'color_salt_paper']
        noises: list[str] | None = list()
        for noise in noise_types:
            prob = 0.5
            if noise is None:
                noises = noise
            else:
                noises.append(noise)
            noise_augmenter = NoiseAugmenter(noise_types=noises, prob=prob)
            dataset = SRDataset(lr_dir, hr_dir, transform=transform, noise_augmenter=noise_augmenter)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for self.epoch in range(1, self.epochs + 1):
                pbar = tqdm(
                    dataloader,
                    unit='bath',
                    ncols=lcolumn,
                    ascii=True,
                    bar_format='{n}/{total} {l_bar}{bar}| {elapsed}/{remaining} |{rate_noinv_fmt}',
                    desc=f'| {'Noise ' + ', '.join(noises) if noises is not None else 'Not noise'} | Epoch {self.epoch}/{self.epochs} | Loss {running_loss / len(dataloader.dataset):.2f}'
                )
                for lr_imgs, hr_imgs in pbar:
                    lr_imgs = lr_imgs.to(device)
                    hr_imgs = hr_imgs.to(device)
                    self.optimizer.zero_grad()
                    outputs = self.model(lr_imgs)
                    loss = self.criterion(outputs, hr_imgs)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item() * lr_imgs.size(0)
                    pbar.set_description(
                        f'| {'Noise ' + ', '.join(noises) if noises is not None else 'Not noise'} | Epoch {self.epoch}/{self.epochs} | Loss {running_loss / len(dataloader.dataset):.2f}')
                    del lr_imgs, hr_imgs, loss
                    torch.cuda.empty_cache()
                # Периодически проверяем качество модели на валидации
                if self.epoch % 10 == 0 or self.epoch == self.epochs - 1:
                    self.validate_model(dataloader)
                    prob += 0.025
                    noise_augmenter = NoiseAugmenter(noise_types=noises, prob=prob)
                    dataset = SRDataset(lr_dir, hr_dir, transform=transform, noise_augmenter=noise_augmenter)
                    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

                # Сохраняем лучшую модель по PSNR
                if self.avg_psnr > self.best_psnr:
                    self.best_psnr = self.avg_psnr
                    self.model.eval()
                    enhance_image(self.model, f'trained_upscale_model_{self.best_psnr:.4f}')
                    torch.save(self.model.state_dict(), f'{model_dir}/trained_upscale_model_{best_psnr:.4f}.pth')
                torch.cuda.empty_cache()
