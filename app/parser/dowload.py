import uuid
from io import BytesIO

import aiohttp
from PIL import Image, UnidentifiedImageError
from app.config import lcolumn, scale

from tqdm import tqdm


class ImgDownload:
    def __init__(self, data: list[str]):
        self.data: list[str] = data
        self.session = None

    async def refresh_session(self) -> None:
        if self.session is not None:
            await self.session.close()
        self.session = aiohttp.ClientSession()

    async def download(self, file_path: str):
        pbar = tqdm(
            self.data, ascii=True,
            desc='Dowloading files',
            unit='img',
            ncols=lcolumn,
            bar_format='{l_bar}{bar}| {elapsed}/{remaining} | {rate_noinv_fmt}'
        )
        for _idx, item in enumerate(pbar):
            indx = str(uuid.uuid4().hex)
            await self.refresh_session()
            async with self.session.get(item) as response:
                try:
                    if 'лимит скачиваний исчерпан' in await response.text(encoding='UTF-8'):
                        continue
                except:
                    pass
                file = BytesIO(await response.read())
                try:
                    image = Image.open(file)
                    image.save(f'{file_path}/train_high/img_{indx}.jpg')
                    low_image = image.resize((image.width // scale, image.height // scale), Image.NEAREST)
                    low_image.save(f'{file_path}/train_low/img_{indx}.jpg')
                    if _idx % 10:
                        await self.refresh_session()
                except UnidentifiedImageError:
                    pass
            await self.session.close()
