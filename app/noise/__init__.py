import numpy
from PIL import Image
from PIL.ImageFile import ImageFile
from .noises import gaus_noise, salt_a_paper, color_salt_paper, quantize_image, shot_noise, pixelated
from .colors import adjust_brightness, adjust_rgb_channels, adjust_contrast, enhance_sharpness


class NoiseAugmenter(object):
    def __init__(self, noise_types:list[str] | None, prob:float = 0.5):
        self.noise_types: list[str] = noise_types
        self.prob: float = prob

    def add_noise(self, img: ImageFile):
        cv_img = numpy.array(img)
        noisy_img = cv_img.copy()
        if self.noise_types is not None:
            for noise_type in self.noise_types:
                match noise_type:
                    case 'gaus':
                        noisy_img = gaus_noise(noisy_img, self.prob)
                    case 'salt_paper':
                        noisy_img = salt_a_paper(noisy_img, self.prob)
                    case 'color_salt_paper':
                        noisy_img = color_salt_paper(noisy_img, self.prob)
                    case 'quantize':
                        noisy_img = quantize_image(noisy_img, self.prob)
                    case 'shot_noise':
                        noisy_img = shot_noise(noisy_img, self.prob)
                    case 'pixelated':
                        noisy_img = pixelated(noisy_img, self.prob)
                    case 'add_color':
                        noisy_img = adjust_brightness(noisy_img)
                        noisy_img = adjust_rgb_channels(noisy_img)
                        noisy_img = adjust_contrast(noisy_img)
                    case _:
                        ...
        return Image.fromarray(noisy_img)
