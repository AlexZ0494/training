import asyncio

import torch
import torch.nn as nn
import torch.distributed as dist
from app.config import lcolumn
from numba.parfors.parfor import max_checker
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from app.config import device, checkpoin_dir
from app.parser.dowload import ImgDownload
from app.residual_block.test import enhance_image
from app.residual_block.training import TrainModel
from app.residual_block.upscale import UpscaleModel
from app.utils.consolegui import print_center, display_gpu_info
from app.parser.wallpaperscraft import Parse as wallpaperscraft
from app.parser.forkwallpapers import Parse as forkwallpapers
from app.parser.hdqwalls import Parse as hdqwalls

import os

from app.utils.extract_num import extract_number, extract_check


os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ['CUDNN_DETERMINISTIC'] = '1'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision('medium') # или 'high', но не 'highest' при проблемах



if __name__ == "__main__":
    print_center("Dowload images for training model")
    data: list[str] = list()
    data.extend(wallpaperscraft().download_images)
    data.extend(forkwallpapers().download_images)
    data.extend(hdqwalls().download_images)
    ImgDownload(data).download(ind=0)
    # torch.cuda.set_device(1)
    # print_center("Run training model")
    display_gpu_info(torch)
    max_checker: str = ''
    model = UpscaleModel().to(device)
    if len(os.listdir(checkpoin_dir)) > 0:
        max_checker = extract_check(checkpoin_dir)
        checkpoint = torch.load(f'{checkpoin_dir}/{max_checker}')
        model.load_state_dict(checkpoint)
    criterion = nn.MSELoss()
    model_jit = torch.jit.script(model)
    optimizer = torch.optim.Adam(model_jit.parameters(), lr=1e-6)
    TrainModel(
        model,
        criterion,
        optimizer,
        best_psnr=extract_number(max_checker) if max_checker != '' else 0.0
    ).train_model()
    # checkpoint = torch.load(f'app/models/model/checkpoint/checkpoint_14.3417.pth')
    # model.load_state_dict(checkpoint)
    # enhance_image(model, 'checkpoint_14.3417.pth')
