import shutil

import torch

lcolumn: int = shutil.get_terminal_size()[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr_dir: str = 'app/models/dataset/train_low'
hr_dir: str = 'app/models/dataset/train_high'
model_dir: str = 'app/models/model/'
download_path: str = 'app/models/dataset'
noise_types: list[str] = ['gaus', 'color_salt_paper']
prob: float = 0.5
epochs: int = 100
scale: int = 4
batch_size: int = 2
