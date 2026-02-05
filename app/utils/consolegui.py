from typing import Any

from app.config import lcolumn

def display_gpu_info(torch: Any):
    info = [
        ("PyTorch Version", torch.__version__),
        ("CUDA Available", str(torch.cuda.is_available())),
        ("Current Device", str(torch.cuda.current_device())),
        ("Device Count", str(torch.cuda.device_count())),
        ("Device Name", torch.cuda.get_device_name(torch.cuda.current_device()))
    ]

    max_key_len = max(len(key) for key, value in info)
    print('-' * lcolumn)
    for key, value in info:
        print(f"- {key}:{' ' * (max_key_len - len(key))} {value}")
    print('-' * lcolumn)


def print_center(text: str):
    print(f' {text} '.center(lcolumn, '*'))
