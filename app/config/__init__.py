import shutil

import torch

lcolumn: int = shutil.get_terminal_size()[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr_dir: str = 'app/models/dataset/train_low'
hr_dir: str = 'app/models/dataset/train_high'
model_dir: str = 'app/models/model/'
download_path: str = 'app/models/dataset'
