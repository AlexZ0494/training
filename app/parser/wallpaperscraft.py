import aiohttp
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from app.parser.dowload import ImgDownload


class Parse:
    def __init__(self, url: str = 'https://wallpaperscraft.ru/all/1920x1080'):
        self.url: str = url
        html = requests.get(url).text
        self.soup = BeautifulSoup(html, 'lxml')

    async def download_images(self, file_path: str) -> None:
        items: list[str] = list()
        max_page: int = int(self.soup.select('.pager__item_last-page a')[0].get('href').replace('/all/1920x1080/page', ''))
        pbar = tqdm(range(1, max_page), ascii=True, unit='page')
        async with aiohttp.ClientSession() as session:
            for i in pbar:
                if i == 1:
                    url = self.url
                else:
                    url = self.url + f'/page{i}'
                pbar.set_description(f"Parsing page '{url}'")
                async with session.get(url) as response:
                    page = BeautifulSoup(await response.text(encoding='UTF-8'), 'lxml')
                    for image in page.select('.wallpapers__link img'):
                        items.append(image.get('src').replace('300x168', '1920x1080'))
        await ImgDownload(items).download(file_path)
