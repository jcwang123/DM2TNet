import sys, os, argparse
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.Generator import *
from utils.metrics import *
from utils.pre import *
from utils.summary import create_logger, create_summary
from torch_network.vnet3d import VNet

import medpy.metric as md

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--arch', type=str, default='dmmtnet')
    parser.add_argument('--dataset', type=str, default='p20')

    parse_config = parser.parse_args()
    print(parse_config)

    if parse_config.dataset == 'mosmed':
        parse_config.data_path = '/raid/wjc/data/mosmed'
    elif parse_config.dataset == 'p20':
        parse_config.data_path = '/raid/wjc/data/covid-19-p20/ori_data/'

    os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu

    exp_name = parse_config.dataset + '/' + parse_config.arch + '/overall'

    parse_config.txt_dir = './logs/' + exp_name + '/txts'
    parse_config.img_dir = './logs/' + exp_name + '/imgs'

    os.makedirs(parse_config.img_dir, exist_ok=True)
    os.makedirs(parse_config.txt_dir, exist_ok=True)
    logger = create_logger(0, save_dir=parse_config.txt_dir)
    print = logger.info
    print(parse_config)

    if parse_config.arch == 'unetpp':
        from segmentation_models_pytorch import UnetPlusPlus
        model = UnetPlusPlus(in_channels=1, classes=1,
                             encoder_weights=None).cuda()
        tag = '2d'
        log_dir = 'logs/mosmed/unetpp'
        cube_size = (256, 256)
    elif parse_config.arch == 'unet':
        from segmentation_models_pytorch import Unet
        model = Unet(in_channels=1, classes=1, encoder_weights=None).cuda()
        tag = '2d'
        log_dir = 'logs/mosmed/unet'
        cube_size = (256, 256)
    elif parse_config.arch == 'nnunet':
        from torch_network.nn_unet3d import nnUNet3D
        cube_size = (160, 160, 32)
        model = nnUNet3D(1).cuda()
        log_dir = 'logs/mosmed/nnunet'
        output = 'single'
        tag = '3d'
    elif parse_config.arch == 'dmmtnet':
        from torch_network.dmmt import DMMTNet
        cube_size = (160, 160, 32)
        model = DMMTNet(1, outputs='multi').cuda()
        log_dir = 'logs/mosmed/dmmtnet_multi_mt_0.1'
        output = 'multi'
        tag = '3d'
    else:
        raise NotImplementedError

    from utils.SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance

    def nsd(pre, label):
        surface_distances = compute_surface_distances(label, pre, [1, 1, 8])
        v = compute_surface_dice_at_tolerance(surface_distances, 8)
        return v

    eval_funcs = [md.dc, md.jc, nsd, md.asd, md.hd95]
    eval_names = ['dice', 'jaccard', 'nsd', 'asd', 'hd95']
    scores = []
    print('[INFO] Start testing...')

    test_images = glob.glob(parse_config.data_path + '/ct_scan/*.nii')
    test_labels = glob.glob(parse_config.data_path + '/infection_mask/*.nii')
    test_images.sort()
    test_labels.sort()

    tic = 0

    fold = 0
    model_dir = os.path.join(log_dir,
                             'fold_{}/models/best_model.t7'.format(fold))
    model.load_state_dict(torch.load(model_dir))

    model.eval()
    dices = []

    for i in range(len(test_images)):
        new_spacing = [1.62, 1.62, 8]
        img, origin, spacing, direction = load_image(test_images[i])
        labels, origin, spacing2, direction = load_image(test_labels[i])
        context = test_images[i].split('/')[-1][:-4]

        if context == 'radiopaedia_org_covid-19-pneumonia-29_86490_1-dcm':
            continue
        if context[0] == 'c':
            pass
        else:
            img = img[:, ::-1]
            labels = labels[:, ::-1]

        # nomalize the scale
        shape_x, shape_y, shape_z = img.shape
        scale = np.asarray([
            spacing[0] / new_spacing[0], spacing[1] / new_spacing[1],
            spacing[2] / new_spacing[2]
        ])
        old_shape = np.asarray([shape_x, shape_y, shape_z])
        new_shape = (old_shape * scale).astype('int')
        new_shape = [int(new_shape[0]), int(new_shape[1]), int(new_shape[2])]

        new_img, new_label = pair_zoom(img, labels, new_shape, new_spacing,
                                       spacing, None, None)

        with torch.no_grad():
            if tag == '2d':
                from test_2d import predict_sample_2d
                prediction = predict_sample_2d(img, model, cube_size)
            elif tag == '3d':
                from test import predict_sample
                prediction = predict_sample(img,
                                            model,
                                            cube_size,
                                            output=output)[0]
        score = []

        for i in range(len(eval_funcs)):
            eval_func = eval_funcs[i]
            v = eval_func(prediction, labels > 0.5)
            if i <= 2:
                v *= 100
            score.append(v)
            context += ' {}:{:.2f}'.format(eval_names[i], score[i])

        print(context)
        # import matplotlib.pyplot as plt
        # plt.imsave('1.jpg', img[..., 10])
        # sys.exit()
        scores.append(score)

    scores = np.array(scores)
    avg_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)

    context = ''
    for i in range(len(eval_funcs)):
        context += '&${:.2f}\pm{:.2f}$'.format(avg_scores[i], std_scores[i])
    context += '\\\\'
    print(context)

    avg_scores = np.mean(scores[:10], axis=0)
    std_scores = np.std(scores[:10], axis=0)

    context = ''
    for i in range(len(eval_funcs)):
        context += '&${:.2f}\pm{:.2f}$'.format(avg_scores[i], std_scores[i])
    context += '\\\\'
    print(context)

    avg_scores = np.mean(scores[10:], axis=0)
    std_scores = np.std(scores[10:], axis=0)

    context = ''
    for i in range(len(eval_funcs)):
        context += '&${:.2f}\pm{:.2f}$'.format(avg_scores[i], std_scores[i])
    context += '\\\\'
    print(context)