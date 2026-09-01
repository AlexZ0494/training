import forkwallpapers, hdqwalls, wallpaperscraft


async def download_images(file_path: str = '/app/models/dataset'):

    await forkwallpapers.Parse().download_images(file_path)
    await hdqwalls.Parse().download_images(file_path)
    await wallpaperscraft.Parse().download_images(file_path)
