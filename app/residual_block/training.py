import datetime
import math

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_skimage

from app.config import device, lcolumn, model_dir, lr_dir, hr_dir, checkpoin_dir
from app.models.dataset import SRDataset
from app.noise import NoiseAugmenter
from app.utils.consolegui import print_center


class TrainModel:
    def __init__(self, model, criterion, optimizer, batch_size: int = 5, best_psnr: float = 0.0):
        self.model = model.to(device)
        self.batch_size = batch_size
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.epoch: int = 1
        self.avg_psnr: float = float('inf')
        self.best_psnr: float = best_psnr
        self.version: float = 0.01
        self.total_ssim: float = 0.0

    def validate_model(self, dataloader, save_check: bool = False, exit_training: bool = False) -> None:
        print_center(
            f"Checkpoint: {self.epoch}" if save_check is False and exit_training is False else "Validate Checkpoint" if exit_training is False else "Exit Trainig"
        )
        self.model.eval()

        # Сначала получаем ссылку на dataset, который использовался для формирования dataloader
        dataset = dataloader.dataset

        # Теперь можно корректно использовать random_split на dataset
        train_size: int = int(len(dataset) * 0.8)
        val_size: int = len(dataset) - train_size
        _, val_dataset = random_split(dataset, [train_size, val_size])

        # Формируем новый валидный DataLoader
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        total_psnr: float = 0.0
        total_ssim: float = 0.0
        count: int = 0

        with torch.no_grad():
            for lr_imgs, hr_imgs in val_dataloader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                outputs = self.model(lr_imgs)

                # Рассчитываем PSNR
                psnr_val: float = float(20 * math.log10(1.0 / math.sqrt(torch.mean((outputs - hr_imgs) ** 2))))
                total_psnr += psnr_val

                # Рассчитываем SSIM
                try:
                    ssim_val = ssim_skimage(
                        outputs.float().detach().cpu().numpy(),
                        hr_imgs.float().detach().cpu().numpy(),
                        multichannel=True,
                        data_range=1.0,
                        win_size=11
                    )
                except:
                    ssim_val = 0.0
                self.total_ssim += ssim_val

                count += 1

        self.avg_psnr = total_psnr / count
        avg_ssim: float = self.total_ssim / count

        print("-" * lcolumn)
        print(f"Average PSNR: {self.avg_psnr:.4f}")
        print(f"Best PSNR: {self.best_psnr:.4f}" if save_check is False else f"Best PSNR: 0")  # Предполагается, что self.best_psnr инициализирован ранее
        print(f"Average SSIM: {avg_ssim:.4f}")
        print("-" * lcolumn)
        print("*" * lcolumn)

        # Сохраняем лучшую модель по PSNR
        if self.avg_psnr > self.best_psnr and save_check is False:
            self.best_psnr = self.avg_psnr
            torch.save(self.model.state_dict(), f'{checkpoin_dir}/checkpoint_{self.best_psnr:.4f}.pth')
        else:
            self.best_psnr = self.avg_psnr
        if avg_ssim > 0:
            torch.save(
                self.model.state_dict(),
                f'{model_dir}/QualityLifter-v{self.version:.2f}_ssim{total_ssim:.4f}.pth'
            )
            self.version += 0.01
        if exit_training is True:
            torch.save(self.model.state_dict(), f'{checkpoin_dir}/checkpoint_{self.best_psnr:.4f}.pth')

        self.model.train()

    def train_model(self):
        day_now: datetime = datetime.datetime.now()
        self.model.train()
        running_loss: float = 0
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        noises: list[str] = ['gaus', 'salt_paper', 'quantize']
        prob: float = 8
        check_count: int = 0
        u_loader_epoch: int = -5
        noise_augmenter = NoiseAugmenter(noise_types=noises, prob=prob)
        dataset = SRDataset(
            lr_dir,
            hr_dir,
            transform=transform,
            noise_augmenter=noise_augmenter,
            cnt_im=20_000
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        print_center("Add noise for model")
        if noises is None or len(noises) == 0:
            noise_str = 'No noises'
        elif len(noises) == 1:
            noise_str = f'{noises[0]}'
        else:
            noise_str = '\n    '.join([f"{i + 1}. {noise}" for i, noise in enumerate(noises)])
        output = f"Noises:\n    {noise_str}"
        print(output)
        if self.best_psnr > 0:
            self.validate_model(dataloader, True)
        print(f"max check Best PSNR: {self.best_psnr:.4f}")
        print_center("START Training")
        print(f' {day_now.strftime('%Y-%m-%d %H:%M:%S')} '.center(lcolumn, '-'))
        try:
            while self.total_ssim < 100 and check_count <= 60:
                pbar = tqdm(
                    dataloader,
                    unit='batch',
                    ncols=lcolumn,
                    ascii=True,
                    bar_format='{n}/{total} {l_bar}{bar}| {elapsed}/{remaining} |{rate_noinv_fmt}',
                    desc=f'| Epoch {self.epoch} | Loss {running_loss / len(dataloader.dataset):.2f}',
                    postfix=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
                scaler = torch.amp.GradScaler('cuda')
                for lr_imgs, hr_imgs in pbar:
                    if day_now.day < datetime.datetime.now().day:
                        day_now = datetime.datetime.now()
                        print(f' {day_now.strftime('%Y-%m-%d %H:%M:%S')} '.center(lcolumn, '-'))
                    self.optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(lr_imgs.to(device))
                        loss = self.criterion(outputs, hr_imgs.to(device))
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    running_loss += loss.item() * lr_imgs.size(0)
                    pbar.set_description(
                        f'| Epoch {self.epoch} | Loss {running_loss / len(dataloader.dataset):.2f}')
                    del lr_imgs, hr_imgs, loss
                    torch.cuda.empty_cache()
                    self.validate_model(dataloader)
                torch.cuda.empty_cache()
                self.epoch += 1
        except KeyboardInterrupt:
            self.validate_model(dataloader, exit_training=True)
            torch.save(self.model.state_dict(), f'{checkpoin_dir}/checkpoint_{self.best_psnr:.4f}.pth')

        print("-" * lcolumn, end='\n\n')
        print_center("END Training")
