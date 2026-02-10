import os

from torchvision import transforms
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm

from app.config import device, lr_tst_dir, hr_tst_dir, lcolumn
from app.utils.consolegui import print_center


def enhance_image(model, model_name: str) -> None:
    transform = transforms.Compose([
        transforms.ToTensor(),  # Преобразует изображение в тензор
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Нормализует тензор
    ])
    model.eval()
    print_center(f'Run test model {model_name}')
    pbar = tqdm(
        sorted(os.listdir(lr_tst_dir)),
        unit='img',
        ncols=lcolumn,
        ascii=True,
        bar_format='{n}/{total} {l_bar}{bar}| {elapsed}/{remaining} |{rate_noinv_fmt}',
        desc=f'| {model_name}'
    )
    os.makedirs(f'{hr_tst_dir}/{model_name}')
    for lr_path in pbar:
        pbar.set_description(f'| {model_name} | {lr_path}')
        input_image = Image.open(os.path.join(lr_tst_dir, lr_path)).convert('RGB')
        input_tensor = transform(input_image).unsqueeze(0)  # Добавляем размер batc-size
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            enhanced_tensor = model(input_tensor)
        enhanced_tensor = enhanced_tensor.clamp(0, 1)
        enhanced_image = transforms.ToPILImage()(enhanced_tensor.squeeze(0).cpu())
        enhanced_image.save(f'{hr_tst_dir}/{model_name}/{lr_path}')
    model.train()
    print_center('-' * lcolumn)
