import shutil

import torch

lcolumn: int = shutil.get_terminal_size()[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr_dir: str = 'app/models/dataset/train_low'
hr_dir: str = 'app/models/dataset/train_high'
lr_tst_dir: str = 'app/models/dataset/test_low'
hr_tst_dir: str = 'app/models/dataset/test_high'
model_dir: str = 'app/models/model/'
download_path: str = 'app/models/dataset'
noise_types: list[str] = list()
prob: float = 0.5
epochs: int = 2000
scale: int = 4
batch_size: int = 3
