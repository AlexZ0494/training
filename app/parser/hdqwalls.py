import multiprocessing
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import tqdm

from app.config import lcolumn


def get_url(url: str):
    """
    Парсит страницу и возвращает список изображений.
    """
    items: list[str] = list()
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(3)
    for item in BeautifulSoup(driver.page_source, 'lxml').select('.wall-resp a img'):
        items.append(
            item.get('src').replace('wallpapers/thumb', 'download').replace('.jpg', '-1920x1080.jpg')
        )
    driver.quit()
    return items


class Parse:
    def __init__(self, url: str = 'https://hdqwalls.com/1920x1080-resolution-wallpapers/page/1'):
        self.url: str = url
        self.options = webdriver.ChromeOptions()
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option('useAutomationExtension', False)
        driver = webdriver.Chrome(options=self.options)
        driver.get(url)
        time.sleep(3)
        self.soup: BeautifulSoup = BeautifulSoup(driver.page_source, 'lxml')
        driver.quit()
    @property
    def download_images(self) -> list[str]:
        items: list[str] = list()
        max_page: int = int(self.soup.select('.pagination a')[-2:][0].text) + 1
        with multiprocessing.Pool(processes=7) as pool, tqdm(
                ncols=lcolumn,
                ascii=True,
                unit='page',
                bar_format='{n}/{total} {l_bar}{bar}| {elapsed}/{remaining} |{rate_noinv_fmt}',
                desc=f'| Parsing site {self.url.split('/')[2]}',
                total=max_page) as pbar:
            for result in pool.imap(get_url, [self.url.replace('page/1', f'page/{i}') for i in range(1, max_page)]):
                for item in result:
                    items.append(item)
                pbar.update()
                pbar.refresh()
        return items
