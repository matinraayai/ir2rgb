###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.pgm', '.PGM',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', 
    '.txt', '.json'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(directory):
    images = []
    assert os.path.isdir(directory), f"{directory} does not exist or is no a valid directory"

    for root, _, f_names in sorted(os.walk(directory)):
        for f_name in f_names:
            if is_image_file(f_name):
                path = os.path.join(root, f_name)
                images.append(path)
    return images


def make_grouped_dataset(directory):
    images = []
    assert os.path.isdir(directory), f"{directory} does not exist or is no a valid directory"
    f_names = sorted(os.walk(directory))
    for f_name in sorted(f_names):
        paths = []
        root = f_name[0]
        for f in sorted(f_name[2]):
            if is_image_file(f):
                paths.append(os.path.join(root, f))
        if len(paths) > 0:
            images.append(paths)
    return images


def check_path_valid(ir_paths, rgb_paths, annotations_paths):
    assert(len(ir_paths) == len(rgb_paths) == len(annotations_paths))
    for a, b, c in zip(ir_paths, rgb_paths, annotations_paths):
        assert(len(a) == len(b) == len(c))


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
