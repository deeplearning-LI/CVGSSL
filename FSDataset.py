import numpy as np
import os
from torchvision.datasets import DatasetFolder
from PIL import Image
import random


def make_ltdataset(directory, class_to_idx, LTfactor=10):
    random.seed(10)
    instances = []
    class_num = len(class_to_idx)
    count = np.zeros((class_num), dtype=np.int)

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            # print(fnames)
            random.shuffle(fnames)
            for fname in fnames:
                if count[class_index] < int(
                        np.floor(len(fnames) * ((1 / LTfactor) ** (1 / (class_num - 1))) ** (class_index))):
                    # print(count[class_index])
                    path = os.path.join(root, fname)
                    item = path, class_index
                    # print(item)
                    instances.append(item)
                    count[class_index] += 1
    return instances, count


def make_fsdataset(directory, class_to_idx, fsn=1):
    instances = []
    class_num = len(class_to_idx)
    count = np.zeros((class_num), dtype=int)

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            random.shuffle(fnames)
            for fname in fnames:
                if count[class_index] < fsn:
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)
                    count[class_index] += 1

    return instances, count


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert("RGB")


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class fsdataset(DatasetFolder):

    def __init__(self, root, factor=0, transform=None, target_transform=None):
        super(fsdataset, self).__init__(root=root, loader=pil_loader, extensions=IMG_EXTENSIONS, transform=transform,
                                        target_transform=target_transform)
        classes, self.class_to_idx = self.find_classes(root)
        if factor > 0:
            self.samples, self.count = make_fsdataset(root, self.class_to_idx, factor)


if __name__ == '__main__':
    random.seed(10)
