import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By

from app.parser.dowload import ImgDownload


class Parse:
    def __init__(self, url: str = 'https://akspic.ru/album/1920x1080'):
        self.url: str = url
        options = webdriver.ChromeOptions()
        self.driver = webdriver.Chrome(options=options)

    async def download_images(self, file_path: str) -> None:
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
                        soups.append(BeautifulSoup(self.driver.page_source, 'lxml').select('.gallery_fluid-column-block'))
                        next_page.click()
                        time.sleep(3)
                except:
                    soups.append(BeautifulSoup(self.driver.page_source, 'lxml').select('.gallery_fluid-column-block'))
                    break
            else:
                last_height = new_height
        self.driver.quit()
        for item in soups:
            items.append(item.get('href'))
        await ImgDownload(items).download(file_path)
