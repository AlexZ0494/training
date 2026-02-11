import multiprocessing
import time

import aiohttp
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import tqdm

from app.config import lcolumn


def get_url(url: str) -> str | None:
    try:
        item = BeautifulSoup(requests.get(url).text, 'lxml').select(
            '.wallpaper__image__desktop img')[0]
        return item.get('src')
    except Exception:
        return None


class Parse:
    def __init__(self, url: str = 'https://akspic.ru/album/1920x1080'):
        self.url: str = url
        options = webdriver.ChromeOptions()
        self.driver = webdriver.Chrome(options=options)

    @property
    async def download_images(self) -> list[str]:
        items: list[str] = list()
        soups: list = list()
        self.driver.get(self.url)
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                try:
                    next_page = self.driver.find_element(By.CSS_SELECTOR, ".navigation .grid .grid__col .showmore a")
                    if str(next_page.text).strip() == "Следующая страница":
                        for block in BeautifulSoup(self.driver.page_source, 'lxml').select(
                                '.gallery_fluid-column-block'):
                            soups.append(block.get('href'))
                        next_page.click()
                        time.sleep(3)
                except:
                    for block in BeautifulSoup(self.driver.page_source, 'lxml').select('.gallery_fluid-column-block'):
                        soups.append(block)
                    break
            else:
                last_height = new_height
        self.driver.quit()

        with multiprocessing.Pool(processes=20) as pool, tqdm(
                ncols=lcolumn,
                ascii=True,
                unit='img',
                bar_format='{n}/{total} {l_bar}{bar}| {elapsed}/{remaining} |{rate_noinv_fmt}',
                desc=f'| Parsing site {self.url.split('/')[2]}',
                total=len(soups)) as pbar:
            for result in pool.imap(get_url, soups):
                if result is not None:
                    items.append(result)
                pbar.update()
                pbar.refresh()

        return items
