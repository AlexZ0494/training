import forkwallpapers, hdqwalls, wallpaperscraft, akspic


async def download_images(file_path: str = '/app/models/dataset'):

    await forkwallpapers.Parse().download_images(file_path)
    await hdqwalls.Parse().download_images(file_path)
    await wallpaperscraft.Parse().download_images(file_path)

    # await akspic.Parse().download_images(file_path)  # требует больше затрат по проходам webdriver
