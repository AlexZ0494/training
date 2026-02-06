import numpy
from PIL import Image
from PIL.ImageFile import ImageFile
from .noises import gaus_noise, salt_a_paper, color_salt_paper, quantize_image, shot_noise
from app.config import noise_types, prob


class NoiseAugmenter:
    def __init__(self):
        self.noises:list[str] = noise_types
        self.prob:float = prob

    def add_noise(self, img: ImageFile):
        cv_img = numpy.array(img)
        noisy_img = cv_img.copy()
        for noise_type in noise_types:
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
                case _:
                    ...
        Image.fromarray(noisy_img).show()
        return Image.fromarray(noisy_img)
