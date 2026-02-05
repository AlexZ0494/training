import re

import aiohttp
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from app.parser.dowload import ImgDownload


class Parse:
    def __init__(self, url: str = 'https://4kwallpapers.com/'):
        self.url: str = url
        html = requests.get(url).text
        self.soup = BeautifulSoup(html, 'lxml')

    async def download_images(self, file_path: str, scale: int = 4) -> None:
        items: list[str] = list()
        max_page: int = int(self.soup.select('.pages a')[-2:][0].text)
        pbar = tqdm(
            range(1, 3),
            ascii=True,
            unit='page',
            ncols=150,
            bar_format='{n}/{total} {l_bar}{bar}| {elapsed}/{remaining} | {rate_noinv_fmt}',
            desc='Parsing pages 4kwallpapers'
        )
        async with aiohttp.ClientSession() as session:
            for i in pbar:
                if i == 1:
                    url = self.url
                else:
                    url = self.url + f'?page={i}'
                async with session.get(url) as response:
                    page = BeautifulSoup(await response.text(encoding='UTF-8'), 'lxml')
                    for image in page.select('.wallpapers__canvas_image'):
                        match = re.search(r'/([^/]+)-(\d+)', image.get('href').replace('.html', ''))
                        if match:
                            name = match.group(1)
                            number = match.group(2)
                            new_link = f'https://4kwallpapers.com/images/wallpapers/{name}-1920x1080-{number}.jpg'
                            items.append(new_link)
        await ImgDownload(items, scale).download(file_path)