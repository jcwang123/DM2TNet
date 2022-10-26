import numpy as np
import SimpleITK as sitk
import cv2, time
from matplotlib import pyplot as plt
import csv, os, random, glob
from .pre import *
from .transform import *


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def generate_npy_sliding(save_dir, size, overlap, sets):
    make_directory(save_dir)
    num = 0
    for _set in sets:
        images, origin, spacing, direction = load_image(_set)
        labels, origin, spacing, direction = load_image(_set)
        imagez = images.shape[-1]
        masks = np.zeros((1, ) + images.shape)
        stepz = int(size[-1] * (1 - overlap))
        step = int(size[0] * (1 - overlap))
        k = 0
        positions = []
        for k in range((images.shape[-1] - size[-1]) // stepz + 1):
            for i in range((images.shape[0] - size[0]) // step + 1):
                for j in range((images.shape[1] - size[1]) // step + 1):
                    positions.append([i * step, j * step, k * stepz])
                positions.append(
                    [i * step, images.shape[1] - size[1], k * stepz])
            for j in range((images.shape[1] - size[1]) // step + 1):
                positions.append(
                    [images.shape[0] - size[0], j * step, k * stepz])
            positions.append([
                images.shape[0] - size[0], images.shape[1] - size[1], k * stepz
            ])
        for i in range((images.shape[0] - size[0]) // step + 1):
            for j in range((images.shape[1] - size[1]) // step + 1):
                positions.append(
                    [i * step, j * step, images.shape[-1] - size[-1]])
            positions.append([
                i * step, images.shape[1] - size[1],
                images.shape[-1] - size[-1]
            ])
        for j in range((images.shape[1] - size[1]) // step + 1):
            positions.append([
                images.shape[0] - size[0], j * step,
                images.shape[-1] - size[-1]
            ])
        positions.append([
            images.shape[0] - size[0], images.shape[1] - size[1],
            images.shape[-1] - size[-1]
        ])
        for position in positions:
            img = normalize(images[position[0]:position[0] + size[0],
                                   position[1]:position[1] + size[1],
                                   position[2]:position[2] + size[2]])
            rate = np.sum(img > 0) / (img.shape[0] * img.shape[1] *
                                      img.shape[2])
            if rate > 0.2:
                num += 1
                label = labels[position[0]:position[0] + size[0],
                               position[1]:position[1] + size[1],
                               position[2]:position[2] + size[2]]
                np.save(os.path.join(save_dir, 'image_{:5d}.npy'.format(num)),
                        image)
                np.save(os.path.join(save_dir, 'label_{:5d}.npy'.format(num)),
                        label)
        print(_set, len(positions), num)


def generate_npy_random(save_dir, size, overlap, sets, number):
    make_directory(save_dir)
    for k, _set in enumerate(sets):
        images, origin, spacing, direction = load_image(_set)
        labels, origin, spacing, direction = load_image(_set)
        imagez = images.shape[-1]
        masks = np.zeros((1, ) + images.shape)
        num = 0
        while num < number:
            position = [
                random.randint(0, images.shape[0] - size[0]),
                random.randint(0, images.shape[1] - size[1]),
                random.randint(0, images.shape[2] - size[2])
            ]
            img = normalize(images[position[0]:position[0] + size[0],
                                   position[1]:position[1] + size[1],
                                   position[2]:position[2] + size[2]])
            rate = np.sum(img > 0) / (img.shape[0] * img.shape[1] *
                                      img.shape[2])

            if rate > 0.3:
                num += 1
                label = labels[position[0]:position[0] + size[0],
                               position[1]:position[1] + size[1],
                               position[2]:position[2] + size[2]]
                np.save(
                    os.path.join(save_dir,
                                 'image_{:5d}.npy'.format(num + k * number)),
                    image)
                np.save(
                    os.path.join(save_dir,
                                 'label_{:5d}.npy'.format(num + k * number)),
                    label)
