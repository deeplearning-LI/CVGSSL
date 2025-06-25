from PIL import ImageFilter,ImageOps
import random


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]

class MutilCropsTransform:
    """Take Mutil random crops of one image as the query and key.
        n_views=mutil"""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)