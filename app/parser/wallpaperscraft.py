import multiprocessing
import queue
import time

import aiohttp
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from app.parser.dowload import ImgDownload
from app.config import lcolumn
from app.utils.consolegui import print_center


def get_url(url: str):
    """
    Парсит страницу и возвращает список изображений.
    """
    items: list[str] = list()
    run = True
    while run:
        try:
            page = BeautifulSoup(requests.get(url).text, 'lxml')
            for image in page.select('.wallpapers__link img'):
                items.append(image.get('src').replace('300x168', '1920x1080'))
            run = False
        except requests.exceptions.ConnectionError:
            time.sleep(75)
    return items


class Parse:
    def __init__(self, url: str = 'https://wallpaperscraft.ru/all/1920x1080'):
        self.url: str = url
        html = requests.get(url).text
        self.soup = BeautifulSoup(html, 'lxml')

    async def download_images(self) -> None:
        """
        Скачивает изображения с сайта, используя многопоточность и прогрессбар.
        :param file_path: путь для сохранения изображений
        """
        items: list[str] = list()
        max_page: int = int(
            self.soup.select('.pager__item_last-page a')[0].get('href').replace('/all/1920x1080/page', ''))
        with multiprocessing.Pool(processes=11) as pool, tqdm(
                ncols=lcolumn,
                ascii=True,
                unit='page',
                bar_format='{n}/{total} {l_bar}{bar}| {elapsed}/{remaining} |{rate_noinv_fmt}',
                desc=f'| Parsing site {self.url.split('/')[2]}',
                total=max_page) as pbar:
            for result in pool.imap(get_url, [self.url + f'/page{i}' if i != 1 else self.url for i in range(1, max_page)]):
                for item in result:
                    items.append(item)
                pbar.update()
                pbar.refresh()

        await ImgDownload(items).download()
