import numpy as np
import random, glob, cv2
from .pre import *
from .transform import *

import torch
import torch.utils.data


def load_division(fold):
    import json
    with open('config/mosmed.json', 'r') as f:
        division = json.load(f)
    train_ids, test_ids = [], []
    for f in range(5):
        if int(fold) == f:
            test_ids += division[str(f)]
        else:
            train_ids += division[str(f)]
    return train_ids, test_ids


def pooling(cube, rate):
    #b*c*x*y*z
    batch = cube.shape[0]
    channel = cube.shape[1]
    width = cube.shape[2]
    height = cube.shape[3]
    slices = cube.shape[4]

    pool_out = np.zeros(
        (batch, channel, width // rate, height // rate, slices // rate),
        dtype=np.uint8)
    stride = rate
    for b in range(batch):
        for x in np.arange(0, width, stride):
            for y in np.arange(0, height, stride):
                for z in np.arange(0, slices, stride):
                    pool_out[b, 0, x // rate, y // rate,
                             z // rate] = np.max(cube[b, 0, x:x + stride,
                                                      y:y + stride,
                                                      z:z + stride])
    return pool_out


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def unlabel_sample(image_path, sample_size):
    img, origin, spacing, direction = load_image(image_path)
    ori_shape = img.shape
    if ori_shape[2] - sample_size[2] >= 0:
        sample_position = [
            random.randint(0, ori_shape[0] - sample_size[0]),
            random.randint(0, ori_shape[1] - sample_size[1]),
            random.randint(0, ori_shape[2] - sample_size[2])
        ]
        new_img = img[sample_position[0]:(sample_position[0] + sample_size[0]),
                      sample_position[1]:sample_position[1] + sample_size[1],
                      sample_position[2]:sample_position[2] + sample_size[2]]
    else:
        new_img = np.zeros(sample_size)
        sample_position = [
            random.randint(0, ori_shape[0] - sample_size[0]),
            random.randint(0, ori_shape[1] - sample_size[1])
        ]
        new_img[:, :, :ori_shape[2]] = img[sample_position[0]:(
            sample_position[0] +
            sample_size[0]), sample_position[1]:sample_position[1] +
                                           sample_size[1], :]

    return new_img


def sample_2d(image_path, anno_path, sample_size=(256, 256)):
    img, origin, spacing, direction = load_image(image_path)
    label, origin, spacing, direction = load_image(anno_path)
    sample_z = random.randint(0, img.shape[-1] - 1)
    img = img[..., sample_z].copy()
    label = label[..., sample_z].copy()
    if not img.shape == sample_size:
        img = cv2.resize(img, sample_size)
        label = cv2.resize(label, sample_size)

    return img, label


def sample(image_path, anno_path, sample_size):
    img, origin, spacing, direction = load_image(image_path)
    ori_shape = img.shape
    sample_position = [
        random.randint(0, ori_shape[0] - sample_size[0]),
        random.randint(0, ori_shape[1] - sample_size[1]),
        random.randint(0, ori_shape[2] - sample_size[2])
    ]
    new_img = img[sample_position[0]:(sample_position[0] + sample_size[0]),
                  sample_position[1]:sample_position[1] + sample_size[1],
                  sample_position[2]:sample_position[2] + sample_size[2]]
    if anno_path == 'normal':
        new_label = np.zeros(new_img.shape)
    else:
        label, origin, spacing, direction = load_image(anno_path)
        new_label = label[sample_position[0]:sample_position[0] +
                          sample_size[0],
                          sample_position[1]:sample_position[1] +
                          sample_size[1],
                          sample_position[2]:sample_position[2] +
                          sample_size[2]]
    return new_img, new_label


class mosmed2d(torch.utils.data.Dataset):
    def __init__(self, image_paths, anno_paths, sample_size, bs):
        super(mosmed2d, self).__init__()
        self.image_paths = image_paths
        self.anno_paths = anno_paths
        self.sample_size = sample_size
        self.bs = bs

    def __len__(self):
        return 100 * self.bs

    def __getitem__(self, index):
        n = len(self.image_paths)
        index = random.randint(0, n - 1)

        image_path = self.image_paths[index]
        anno_path = self.anno_paths[index]

        img, label = sample_2d(image_path,
                               anno_path,
                               sample_size=self.sample_size)

        img = normalize(img)

        return torch.from_numpy(img), torch.from_numpy(label)


class mosmed(torch.utils.data.Dataset):
    def __init__(self, image_paths, anno_paths, sample_size, output, bs):
        super(mosmed, self).__init__()
        self.image_paths = image_paths
        self.anno_paths = anno_paths
        self.sample_size = sample_size
        self.bs = bs

    def __len__(self):
        return 100 * self.bs

    def __getitem__(self, index):
        n = len(self.image_paths)
        index = random.randint(0, n - 1)

        image_path = self.image_paths[index]
        anno_path = self.anno_paths[index]

        img, label = sample(image_path,
                            anno_path,
                            sample_size=self.sample_size)
        img = add_noise(img, 10)
        img = normalize(img)

        return torch.from_numpy(img), torch.from_numpy(label)


class UnlabelDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, sample_size, bs):
        super(UnlabelDataset, self).__init__()
        self.image_paths = image_paths
        self.sample_size = sample_size
        self.bs = bs

    def __len__(self):
        return 100 * self.bs

    def __getitem__(self, index):
        n = len(self.image_paths)
        index = random.randint(0, n - 1)

        image_path = self.image_paths[index]

        img = unlabel_sample(image_path, sample_size=self.sample_size)
        img = add_noise(img, 10)
        img = normalize(img)

        return torch.from_numpy(img)