import asyncio
import math

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.config import device, lr_dir, hr_dir, model_dir, epochs, batch_size, prob, noise_types
from app.residual_block.training import TrainModel
from app.residual_block.upscale import UpscaleModel
from app.models.dataset import SRDataset
from app.utils.consolegui import display_gpu_info, print_center, lcolumn
from app.parser.wallpaperscraft import Parse as wallpaperscraft
from app.parser.forkwallpapers import Parse as forkwallpapers
from app.parser.akspic import Parse as akspic


import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


if __name__ == "__main__":
    print_center("Dowload images for training model")
    download_path: str = 'app/models/dataset'
    asyncio.run(wallpaperscraft().download_images())
    asyncio.run(forkwallpapers().download_images())
    # asyncio.run(akspic().download_images(download_path))
    # # torch.cuda.set_device(1)
    # print_center("Run training model")
    # display_gpu_info(torch)
    #
    # model = UpscaleModel().to(device)
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    #
    # TrainModel(model, criterion, optimizer).train_model()
