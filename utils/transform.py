import numpy as np
import random


def flip_2d(image, label):
    flip_x = random.random() > 0.5
    flip_y = random.random() > 0.5
    if flip_x:
        image = image[::-1]
        label = label[::-1]
    if flip_y:
        image = image[:, ::-1]
        label = label[:, ::-1]
    return image, label


def add_noise_2d(image, sigma):
    noise_image = image + np.random.randn(
        image.shape[0], image.shape[1]) * random.randint(0, sigma) / 255
    return noise_image


def flip(image, label):
    flip_x = random.random() > 0.5
    flip_y = random.random() > 0.5
    flip_z = random.random() > 0.5
    if flip_x:
        image = image[::-1]
        label = label[::-1]
    if flip_y:
        image = image[:, ::-1]
        label = label[:, ::-1]
    if flip_z:
        image = image[..., ::-1]
        label = label[..., ::-1]
    return image, label


def add_noise(image, sigma):
    noise_image = image + np.random.randn(image.shape[0], image.shape[1],
                                          image.shape[2]) * random.randint(
                                              0, sigma) / 255
    return noise_image