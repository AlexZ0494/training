import datetime
import time
import math

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_skimage

from app.config import device, lcolumn, model_dir, lr_dir, hr_dir, checkpoin_dir, lv_dir, hv_dir
from app.models.dataset import SRDataset
from app.noise import NoiseAugmenter
from app.utils.consolegui import print_center


class TrainModel:
    def __init__(self, model, criterion, optimizer, batch_size: int = 6, best_psnr: float = 0.0):
        self.model = model.to(device)
        self.batch_size = batch_size
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.epoch: int = 1
        self.best_psnr: float = best_psnr
        self.version: float = 0.1
        self.total_ssim: float = 0.0
        self.avg_ssim: float = 0.0
        self.check_count: int = 0
        self.noises: list[str] = ['gaus', 'salt_paper', 'quantize', 'color_salt_paper']
        self.noises_test: list[str] = list()

    def validate_model(self, save_check: bool = False, exit_training: bool = False) -> None:
        start_dt: datetime = datetime.datetime.now()
        print_center(
            f"Checkpoint: {self.epoch}" if save_check is False and exit_training is False else "Validate Checkpoint" if exit_training is False else "Exit Trainig"
        )
        self.model.eval()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        noise_augmenter = NoiseAugmenter(self.noises_test, 16)
        val_dataset = SRDataset(
            lv_dir,
            hv_dir,
            transform=transform,
            cnt_im_start=300,
            noise_augmenter=noise_augmenter
        )

        # 2. Оборачиваем его в DataLoader с тем же batch_size
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size // 2,  # Используем тот же размер батча, что и при обучении
            shuffle=False  # Валидацию обычно не перемешивают
        )

        total_psnr: float = 0.0
        total_ssim: float = 0.0
        ssim_val: float = 0.0
        count: int = 0

        pbar = tqdm(
            val_loader,
            unit='obj',
            ncols=lcolumn,
            ascii=True,
            bar_format='{n}/{total} {l_bar}{bar}| {elapsed}/{remaining} |{rate_noinv_fmt}',
            desc=f'| AVG PSNR: {total_psnr:.4f} | AVG SSIM: {ssim_val:.4f}'
        )

        with torch.no_grad():
            for lr_imgs, hr_imgs in pbar:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                outputs = self.model(lr_imgs)

                # Рассчитываем PSNR
                psnr_val: float = float(20 * math.log10(1.0 / math.sqrt(torch.mean((outputs - hr_imgs) ** 2))))
                total_psnr += psnr_val

                outputs_np = outputs.float().detach().cpu().numpy()
                hr_imgs_np = hr_imgs.float().detach().cpu().numpy()
                # Рассчитываем SSIM
                ssim_val = ssim_skimage(
                    outputs_np,
                    hr_imgs_np,
                    multichannel=True,
                    channel_axis=-1,
                    data_range=255,
                    win_size=3
                )
                self.total_ssim += ssim_val
                count += 1
                pbar.set_description(
                    f'| AVG PSNR: {total_psnr / count:.4f} | AVG SSIM: {self.total_ssim / count:.4f}')
        avg_psnr = total_psnr / count
        self.avg_ssim = self.total_ssim / count if self.total_ssim / count > 0 else 0
        time_spent: datetime = datetime.datetime.now() - start_dt
        print("-" * lcolumn)
        print(f"Average PSNR: {avg_psnr:.4f}")
        print(f"Best PSNR: {self.best_psnr:.4f}" if save_check is False else f"Best PSNR: 0")  # Предполагается, что self.best_psnr инициализирован ранее
        print(f"Average SSIM: {self.avg_ssim:.4f}")
        print(f"Time spent on validation: {time.strftime("%H:%M:%S", time.gmtime(time_spent.seconds))}")
        print("-" * lcolumn)
        print("*" * lcolumn)

        # Сохраняем лучшую модель по PSNR
        if avg_psnr > self.best_psnr and save_check is False:
            self.best_psnr = avg_psnr
            torch.save(self.model.state_dict(), f'{checkpoin_dir}/checkpoint_{self.best_psnr:.4f}.pth')
        else:
            self.best_psnr = avg_psnr
        if self.avg_ssim >= 1.0:
            self.noises_test.append(self.noises[len(self.noises_test)])
            self.check_count += 1
            torch.save(
                self.model.state_dict(),
                f'{model_dir}/QualityLifter-v{self.version:.2f}_avgpsnr{avg_psnr:.4f}.pth'
            )
            self.version += 0.1
        if exit_training is True:
            torch.save(self.model.state_dict(), f'{checkpoin_dir}/checkpoint_{self.best_psnr:.4f}.pth')
        pbar.close()
        self.model.train()

    def train_model(self):
        self.model.train()
        running_loss: float = 0
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        prob: float = 12
        noise_augmenter = NoiseAugmenter(noise_types=self.noises, prob=prob)
        dataset = SRDataset(
            lr_dir,
            hr_dir,
            transform=transform,
            noise_augmenter=noise_augmenter,
            cnt_im_start=12_000,
            cnt_im_end=9_666
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        print_center("Add noise for model")
        if self.noises is None or len(self.noises) == 0:
            noise_str = 'No noises'
        elif len(self.noises) == 1:
            noise_str = f'{self.noises[0]}'
        else:
            noise_str = '\n    '.join([f"{i + 1}. {noise}" for i, noise in enumerate(self.noises)])
        output = f"Noises:\n    {noise_str}"
        print(output)
        if self.best_psnr > 0:
            self.validate_model(save_check=True)
        print(f"max check Best PSNR: {self.best_psnr:.4f}")
        print_center("START Training")
        day_now: datetime = datetime.datetime.now()
        print(f' {day_now.strftime('%Y-%m-%d %H:%M:%S')} '.center(lcolumn, '-'))
        try:
            while self.avg_ssim <= 1.0 or self.check_count <= 10:
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
                        print()
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
                if self.epoch != 1 and self.epoch % 10 == 0:
                    self.validate_model()
                torch.cuda.empty_cache()
                self.epoch += 1
        except KeyboardInterrupt:
            self.validate_model(exit_training=True)
            torch.save(self.model.state_dict(), f'{checkpoin_dir}/checkpoint_{self.best_psnr:.4f}.pth')

        print("-" * lcolumn, end='\n\n')
        print_center("END Training")
