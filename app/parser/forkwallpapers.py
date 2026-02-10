import multiprocessing
import re
import time

import aiohttp
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from app.parser.dowload import ImgDownload
from app.config import lcolumn


def get_url(url: str):
    """
    Парсит страницу и возвращает список изображений.
    """
    items: list[str] = list()
    run = True
    while run:
        try:
            page = BeautifulSoup(requests.get(url).text, 'lxml')
            for image in page.select('.wallpapers__canvas_image'):
                match = re.search(r'/([^/]+)-(\d+)', image.get('href').replace('.html', ''))
                if match:
                    name = match.group(1)
                    number = match.group(2)
                    new_link = f'https://4kwallpapers.com/images/wallpapers/{name}-1920x1080-{number}.jpg'
                    items.append(new_link)
                    run = False
        except requests.exceptions.ConnectionError:
            time.sleep(75)
    return items


class Parse:
    def __init__(self, url: str = 'https://4kwallpapers.com/'):
        self.url: str = url
        html = requests.get(url).text
        self.soup = BeautifulSoup(html, 'lxml')

    async def download_images(self) -> None:
        items: list[str] = list()
        max_page: int = int(self.soup.select('.pages a')[-2:][0].text)
        with multiprocessing.Pool(processes=20) as pool, tqdm(
                ncols=lcolumn,
                ascii=True,
                unit='page',
                bar_format='{n}/{total} {l_bar}{bar}| {elapsed}/{remaining} |{rate_noinv_fmt}',
                desc=f'| Parsing site {self.url.split('/')[2]}',
                total=max_page) as pbar:
            for result in pool.imap(get_url, [self.url + f'?page={i}' if i != 1 else self.url for i in range(1, max_page)]):
                for item in result:
                    items.append(item)
                pbar.update()
                pbar.refresh()

        await ImgDownload(items).download()
