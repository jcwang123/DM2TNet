import torch
from torch.nn import functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import numpy as np


def dsc(x, y, per_class=False):
    if per_class:
        axis = np.arange(len(x.shape) - 1)
    else:
        axis = np.arange(len(x.shape))
    axis = tuple(axis)
    return 2 * np.sum(
        x * y, axis=axis) / (np.sum(x, axis=axis) + np.sum(y, axis=axis))


def compute_iou(x, y):
    interaction = np.sum(x * y)
    union = np.sum((x + y) > 0)
    return interaction / union


def compute_precision(x, y):
    return np.sum(x * y) / np.sum(x)


def compute_recall(x, y):
    return np.sum(x * y) / np.sum(y)


def compute_F1(x, y):
    p = compute_precision(x, y)
    r = compute_recall(x, y)
    return 2 * p * r / (p + r)


def dice_loss(gt, pr, smooth=1e-6):
    """
    compute dice in a batch 
    """
    intersection = torch.sum(gt * pr)
    s = torch.sum(gt) + torch.sum(pr)
    dsc = (intersection * 2 + smooth) / (s + smooth)
    return 1 - dsc


def dice_score(gt, pr, smooth=1e-6):
    """
    compute dice per class
    """
    axis = [0, 2, 3, 4]
    intersection = K.sum(gt * pr, axis=axis)
    s = K.sum(gt, axis=axis) + K.sum(pr, axis=axis)
    dsc = (intersection * 2) / (s)
    return dsc


def dice_score_2d(gt, pr, smooth=1e-6):
    """
    compute dice per class
    """
    axis = [0, 1, 2]
    intersection = K.sum(gt * pr, axis=axis)
    s = K.sum(gt, axis=axis) + K.sum(pr, axis=axis)
    dsc = (intersection * 2) / (s)
    dsc = K.mean(dsc)
    return dsc


# def compute_dice_coefficient_per_instance(mask_gt, mask_pred):
#     """Compute instance soerensen-dice coefficient.

#         compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
#         and the predicted mask `mask_pred` for multiple instances.

#         Args:
#           mask_gt: 3-dim Numpy array of type int. The ground truth image, where 0 means background and 1-N is an
#                    instrument instance.
#           mask_pred: 3-dim Numpy array of type int. The predicted mask, where 0 means background and 1-N is an
#                    instrument instance.

#         Returns:
#           a instance dictionary with the dice coeffcient as float.
#         """
#     # get number of labels in image
#     instances_gt = np.unique(mask_gt)
#     instances_pred = np.unique(mask_pred)

#     # create performance matrix
#     performance_matrix = np.zeros((len(instances_gt), len(instances_pred)))
#     masks = []

#     # calculate dice score for each ground truth to predicted instance
#     for counter_gt, instance_gt in enumerate(instances_gt):

#         # create binary mask for current gt instance
#         gt = mask_gt.copy()
#         gt[mask_gt != instance_gt] = 0
#         gt[mask_gt == instance_gt] = 1

#         masks_row = []
#         for counter_pred, instance_pred in enumerate(instances_pred):
#             # make binary mask for current predicted instance
#             prediction = mask_pred.copy()
#             prediction[mask_pred != instance_pred] = 0
#             prediction[mask_pred == instance_pred] = 1

#             # calculate dice
#             performance_matrix[counter_gt, counter_pred] = compute_dice_coefficient(gt, prediction)
#             masks_row.append([gt, prediction])
#         masks.append(masks_row)

#     # assign instrument instances according to hungarian algorithm
#     label_assignment = hungarian_algorithm(performance_matrix * -1)
#     label_nr_gt, label_nr_pred = label_assignment

#     # get performance per instance
#     image_performance = []
#     for i in range(len(label_nr_gt)):
#         instance_dice = performance_matrix[label_nr_gt[i], label_nr_pred[i]]
#         image_performance.append(instance_dice)

#     n_missing = np.max(len(instances_pred) - len(image_performance), len(image_performance) - len(instances_pred))
#     if n_missing > 0:
#         for i in range(n_missing):
#             image_performance.append(0)

#     output = dict()
#     for i, performance in enumerate(image_performance):
#         if i > 0:
#             output["instrument_{}".format(i - 1)] = performance
#         else:
#             output["background"] = performance

#     return output