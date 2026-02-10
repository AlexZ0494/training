import multiprocessing
import time
import uuid
from io import BytesIO

import aiohttp
import requests
from PIL import Image, UnidentifiedImageError
from app.config import lcolumn, scale, download_path

from tqdm import tqdm


def dwnl(url: str):
    indx = str(uuid.uuid4().hex)
    run = True
    while run:
        try:
            response = requests.get(url)
            try:
                if 'лимит скачиваний исчерпан' in response.text:
                    run = True
                    time.sleep(180)
            except:
                pass
            file = BytesIO(response.content)
            try:
                image = Image.open(file)
                image.save(f'{download_path}/train_high/img_{indx}.jpg')
                low_image = image.resize((image.width // scale, image.height // scale), Image.NEAREST)
                low_image.save(f'{download_path}/train_low/img_{indx}.jpg')
            except UnidentifiedImageError:
                pass
            run = False
        except requests.exceptions.ConnectionError:
            time.sleep(75)


class ImgDownload:
    def __init__(self, data: list[str]):
        self.data: list[str] = data
        self.session = None

    async def refresh_session(self) -> None:
        if self.session is not None:
            await self.session.close()
        self.session = aiohttp.ClientSession()

    async def download(self):
        with multiprocessing.Pool(processes=20) as pool, tqdm(
                ncols=lcolumn,
                ascii=True,
                unit='img',
                bar_format='{l_bar}{bar}| {elapsed}/{remaining} | {rate_noinv_fmt}',
                desc=f'Dowloading files',
                total=len(self.data)) as pbar:
            for _ in pool.imap(dwnl, self.data):
                pbar.update()
                pbar.refresh()
