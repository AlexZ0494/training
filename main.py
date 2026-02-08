import asyncio
import math

import torch
import torch.nn as nn
from pytorch_ssim import ssim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.config import device, lr_dir, hr_dir, model_dir, epochs, batch_size, prob, noise_types
from app.residual_block.upscale import UpscaleModel
from app.models.dataset import SRDataset
from app.utils.consolegui import display_gpu_info, print_center, lcolumn
from app.parser.forkwallpapers import Parse


import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


if __name__ == "__main__":
    # print_center("Dowload images for training model")
    # download_path: str = 'app/models/dataset'
    # asyncio.run(Parse().download_images(download_path))
    torch.cuda.set_device(1)
    print_center("Run training model")
    display_gpu_info(torch)


    model = UpscaleModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    train_model(model, dataloader, criterion, optimizer)
    torch.save(model.state_dict(), f'{model_dir}/trained_upscale_model_last.pth')

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