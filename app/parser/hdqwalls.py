import time

from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import tqdm

from app.parser.dowload import ImgDownload


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

    async def download_images(self, file_path: str) -> None:
        items: list[str] = list()
        max_page: int = int(self.soup.select('.pagination a')[-2:][0].text) + 1
        pbar = tqdm(range(1, max_page), ascii=True, unit='page')
        for i in pbar:
            pbar.set_description(f"Parsing page '{self.url.replace('page/1', f'page/{i}')}'")
            driver = webdriver.Chrome(options=self.options)
            driver.get(self.url.replace('page/1', f'page/{i}'))
            time.sleep(3)
            for item in BeautifulSoup(driver.page_source, 'lxml').select('.wall-resp a img'):
                items.append(
                    item.get('src').replace('wallpapers/thumb', 'download').replace('.jpg', '-1920x1080.jpg')
                )
            driver.quit()
        await ImgDownload(items).download(file_path)
