import uuid
from io import BytesIO

import aiohttp
from PIL import Image, UnidentifiedImageError

from tqdm import tqdm


class ImgDownload:
    def __init__(self, data: list[str], scale: int):
        self.data: list[str] = data
        self.scale: int = scale

    async def download(self, file_path: str):
        pbar = tqdm(
            self.data, ascii=True,
            desc='Dowloading files',
            unit='img',
            ncols=150,
            bar_format='{l_bar}{bar}| {elapsed}/{remaining} | {rate_noinv_fmt}'
        )
        async with aiohttp.ClientSession() as session:
            for item in pbar:
                indx = str(uuid.uuid4().hex)
                async with session.get(item) as response:
                    try:
                        if 'лимит скачиваний исчерпан' in await response.text(encoding='UTF-8'):
                            continue
                    except:
                        pass
                    file = BytesIO(await response.read())
                    try:
                        image = Image.open(file)
                        image.save(f'{file_path}/train_high/img_{indx}.jpg')
                        low_image = image.resize((image.width // self.scale, image.height // self.scale), Image.NEAREST)
                        low_image.save(f'{file_path}/train_low/img_{indx}.jpg')
                    except UnidentifiedImageError:
                        pass
