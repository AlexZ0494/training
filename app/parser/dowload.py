import multiprocessing
import time
from io import BytesIO

import requests
from PIL import Image, UnidentifiedImageError
from app.config import lcolumn, scale, download_path

from tqdm import tqdm


def dwnl(url: tuple[int, str]):
    run = True
    while run:
        try:
            response = requests.get(url[1])
            try:
                if 'лимит скачиваний исчерпан' in response.text:
                    run = True
                    time.sleep(180)
            except:
                pass
            file = BytesIO(response.content)
            try:
                image = Image.open(file).convert('RGB')
                image.save(f'{download_path}/train_high/img_{url[0]}.jpg')
                low_image = image.resize((image.width // scale, image.height // scale), Image.NEAREST)
                low_image.save(f'{download_path}/train_low/img_{url[0]}.jpg')
            except UnidentifiedImageError:
                pass
            run = False
        except requests.exceptions.ConnectionError:
            time.sleep(75)


class ImgDownload:
    def __init__(self, data: list[str]):
        self.data: list[str] = data
        self.session = None

    def download(self):
        with multiprocessing.Pool(processes=20) as pool, tqdm(
                ncols=lcolumn,
                ascii=True,
                unit='img',
                bar_format='{l_bar}{bar}| {elapsed}/{remaining} | {rate_noinv_fmt}',
                desc=f'Dowloading files',
                total=len(self.data)) as pbar:
            for _ in pool.imap(dwnl, enumerate(self.data)):
                pbar.update()
                pbar.refresh()
        print('-' * lcolumn)
