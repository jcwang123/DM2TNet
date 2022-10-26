import csv, glob, time, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

from numpy.lib.npyio import save

from utils.pre import *


def rotatel(x):
    x = x.transpose(1, 0, 2)
    x = x[::-1]
    return x


def rotater(x):
    return x.transpose(1, 0, 2)[:, ::-1]


def covid_19_p20(data_path='/raid/wjc/data/covid-19-p20/'):
    image_paths = glob.glob(data_path + 'ori_data/ct_scan/*.nii')
    label_paths = glob.glob(data_path + 'ori_data/infection_mask/*.nii')
    image_paths.sort()
    label_paths.sort()

    for image_path, label_path in zip(image_paths, label_paths):
        img, origin, spacing, direction = load_image(image_path)
        label, origin, spacing2, direction = load_image(label_path)
        assert spacing[0] == spacing2[0] and spacing[2] == spacing2[2]
        save_path = os.path.join(data_path, 'processed',
                                 os.path.basename(image_path[:-4]))
        print(spacing, img.max(), img.min())
    return


def MosMed(data_path='/raid/wjc/data/mosmed/ori_data/'):
    new_spacing = np.asarray([1.62, 1.62, 8])
    root_path = os.path.join(data_path, '../processed')

    all_paths = []
    for study_id in range(1, 5):
        all_paths += glob.glob(
            os.path.join(data_path, 'studies/CT-{}/*'.format(study_id)))
    print("Find {} CT scans...".format(len(all_paths)))

    # 255~304 contain masks
    def filter_contain_mask(path):
        _id = int(os.path.basename(path).split('_')[1][:-4])
        if _id >= 255 and _id <= 304:
            return True
        else:
            return False

    labeled_paths = filter(filter_contain_mask, all_paths)

    all_paths.sort()
    for path in all_paths:
        if path in labeled_paths:
            base_name = os.path.basename(path)[:-4] + '_mask.nii'
            _id = int(os.path.basename(path).split('_')[1][:-4])

            label_path = os.path.join(data_path, 'masks/', base_name)
            img, origin, spacing, direction = load_image(path)
            label, origin, spacing2, direction = load_image(label_path)
            assert spacing[0] == spacing2[0] and spacing[2] == spacing2[2]
            save_path = os.path.join(root_path, 'labeled')
            img = window_level_normalization(img, 0, 2000)

            # nomalize the scale
            shape_x, shape_y, shape_z = img.shape
            scale = np.asarray([
                spacing[0] / new_spacing[0], spacing[1] / new_spacing[1],
                spacing[2] / new_spacing[2]
            ])
            old_shape = np.asarray([shape_x, shape_y, shape_z])
            new_shape = (old_shape * scale).astype('int')
            new_shape = [
                int(new_shape[0]),
                int(new_shape[1]),
                int(new_shape[2])
            ]

            new_img, new_label = pair_zoom(img, label, new_shape, new_spacing,
                                           spacing, None, None)
            img_save_path = os.path.join(save_path,
                                         '{:04d}/image.nii'.format(_id))
            label_save_path = os.path.join(save_path,
                                           '{:04d}/label.nii'.format(_id))
            make_directory(os.path.dirname(img_save_path))
            save_sitk(new_img, img_save_path, new_spacing, origin, direction)
            save_sitk(new_label, label_save_path, new_spacing, origin,
                      direction)

        else:
            _id = int(os.path.basename(path).split('_')[1][:-4])
            img, origin, spacing, direction = load_image(path)
            img = window_level_normalization(img, 0, 2000)
            save_path = os.path.join(root_path, 'unlabeled')

            # nomalize the scale
            shape_x, shape_y, shape_z = img.shape
            scale = np.asarray([
                spacing[0] / new_spacing[0], spacing[1] / new_spacing[1],
                spacing[2] / new_spacing[2]
            ])
            old_shape = np.asarray([shape_x, shape_y, shape_z])
            new_shape = (old_shape * scale).astype('int')
            new_shape = [
                int(new_shape[0]),
                int(new_shape[1]),
                int(new_shape[2])
            ]
            new_img = zoom(img, new_shape, new_spacing, spacing, None, None)
            img_save_path = os.path.join(save_path,
                                         '{:04d}/image.nii'.format(_id))
            label_save_path = os.path.join(save_path,
                                           '{:04d}/label.nii'.format(_id))
            make_directory(os.path.dirname(img_save_path))
            save_sitk(new_img, img_save_path, new_spacing, origin, direction)

        # break


# def split_data():
#     import random, json
#     labeled_ids = os.listdir('/raid/wjc/data/mosmed/processed/labeled')
#     random.shuffle(labeled_ids)
#     division = {
#         '0': labeled_ids[:10],
#         '1': labeled_ids[10:20],
#         '2': labeled_ids[20:30],
#         '3': labeled_ids[30:40],
#         '4': labeled_ids[40:]
#     }
#     with open('config/mosmed.json', 'w') as f:
#         json.dump(division, f)
#     print(division)

if __name__ == '__main__':
    covid_19_p20()
    # split_data()