import asyncio

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.config import device, lr_dir, hr_dir, download_path, model_dir
from app.models.residual_block.upscale import UpscaleModel
from app.parser.forkwallpapers import Parse
from app.models.dataset import SRDataset
from app.utils.consolegui import display_gpu_info, print_center, lcolumn


def train_model(model, dataloader, criterion, optimizer, epochs: int = 10):
    model.train()
    running_loss: float = 0
    for epoch in range(epochs):
        pbar = tqdm(
            dataloader,
            unit='bath',
            ncols=lcolumn,
            ascii=True,
            bar_format='{n}/{total} {l_bar}{bar}| {elapsed}/{remaining} |{rate_noinv_fmt}',
            desc=f'| Epoch {epoch + 1}/{epochs} | Loss {running_loss / len(dataloader.dataset):.2f}'
        )
        for lr_imgs, hr_imgs in pbar:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * lr_imgs.size(0)
            pbar.set_description(f'| Epoch {epoch + 1}/{epochs} | Loss {running_loss / len(dataloader.dataset):.2f}')
            del lr_imgs, hr_imgs, loss
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # print("Dowload images for training model")
    # download_path: str = 'app/models/dataset'
    # scale = 8
    # asyncio.run(Parse().download_images(download_path, scale))
    print_center("Run training model")
    display_gpu_info(torch)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dataset = SRDataset(lr_dir, hr_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UpscaleModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    train_model(model, dataloader, criterion, optimizer)
    torch.save(model.state_dict(), f'{model_dir}/upscaler_v-00.00.01.pth')

# asyncio.run(Parsewallpaperscraft('https://wallpaperscraft.ru/all/1920x1080').url_images())
# asyncio.run(Parsehdqwalls('https://hdqwalls.com/1920x1080-resolution-wallpapers/page/1').url_images())

# data = asyncio.run(Parsezastavok('https://zastavok.net/').url_images())
# data.extend(asyncio.run(Parseakspic('https://akspic.ru/album/1920x1080').url_images()))

# print(len(data))

# model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
#
# model_path = "RealESRGAN_x4plus.pth"
# upsampler = RealESRGANer(
#             scale=4,
#             model_path=model_path,
#             model=model,
#             tile=0,
#             tile_pad=10,
#             pre_pad=0,
#             half=True)

# image = r"input_image_240x320.jpg"
# scale = 4
#
# img = Image.open(image)
# img = np.array(img)
# output, _ = upsampler.enhance(img, outscale=scale)
#
# output_image = Image.fromarray(output)
# output_image.save('output_image.jpeg')