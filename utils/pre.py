import numpy as np
import os, time, csv
import SimpleITK as sitk


# from preprocess import *
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def window_level_normalization(img, level, window):
    min_HU = level - window / 2
    max_HU = level + window / 2
    img[img > max_HU] = max_HU
    img[img < min_HU] = min_HU
    img = 1. * (img - min_HU) / (max_HU - min_HU)
    return img


def zoom(img, new_shape, new_spacing, old_spacing, origin, direction):

    img = img.swapaxes(0, 2)  #x,y,z -> z,y,x
    sitk_img = sitk.GetImageFromArray(img)
    if origin != None:
        sitk_img.SetOrigin(origin)
    sitk_img.SetSpacing(old_spacing)
    if direction != None:
        sitk_img.SetDirection(direction)

    resample = sitk.ResampleImageFilter()
    if direction != None:
        resample.SetOutputDirection(sitk_img.GetDirection())
    if origin != None:
        resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_shape)
    newimage = resample.Execute(sitk_img)
    return sitk.GetArrayFromImage(newimage).swapaxes(
        0, 2)  # get img z,y,x -> x,y,z


def load_image(fileName):
    itk_img = sitk.ReadImage(fileName)
    img = sitk.GetArrayFromImage(itk_img).swapaxes(0,
                                                   2)  # get img z,y,x -> x,y,z
    origin = itk_img.GetOrigin()
    spacing = itk_img.GetSpacing()
    direction = itk_img.GetDirection()
    return img.astype('float32'), np.asarray(origin), np.asarray(
        spacing), direction


def make_directory(p):
    if not os.path.exists(p):
        os.makedirs(p)


def pair_zoom(img, label, new_shape, new_spacing, old_spacing, origin,
              direction):
    new_img = zoom(img, new_shape, new_spacing, old_spacing, origin, direction)
    new_label = zoom(label, new_shape, new_spacing, old_spacing, origin,
                     direction)
    return new_img, new_label


def save_sitk(img, p, new_spacing, origin, direction):
    sitk_img = sitk.GetImageFromArray(img.swapaxes(0, 2))  #x,y,z -> z,y,x
    sitk_img.SetOrigin(origin)
    sitk_img.SetSpacing(new_spacing)
    sitk_img.SetDirection(direction)
    sitk.WriteImage(sitk_img, p)


def load_sets():
    data_path = '/raid/wjc/data/covid19/'
    f = open(data_path + 'covid19_stage0/metadata.csv', 'r')
    reader = csv.reader(f)
    ct_paths = []
    inf_paths = []
    for i, row in enumerate(reader):
        if i == 0:
            titles = row
            print(titles)
        else:
            ct_p, _, inf_p, _ = row
            ct_paths.append(os.path.basename(ct_p))
            inf_paths.append(os.path.basename(inf_p))
    f.close()
    return ct_paths, inf_paths


def load_folder(indexs):
    #     folder1 = [0,2,5,6,9,10,12,15,16,19]
    #     folder2 = [1,3,4,7,8,11,13,14,17,18]
    image_paths, anno_paths = load_sets()
    ip = []
    ap = []
    for i in range(20):
        if i in indexs:
            ip.append(image_paths[i])
            ap.append(anno_paths[i])
    return ip, ap
