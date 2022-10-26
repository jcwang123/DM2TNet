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


def predict_sample(images, model, size, output='single'):
    masks = np.zeros((1, ) + images.shape)
    stepz = size[-1] // 2
    step = size[0] // 2
    k = 0
    positions = []
    for k in range((images.shape[-1] - size[-1]) // stepz + 1):
        for i in range((images.shape[0] - size[0]) // step + 1):
            for j in range((images.shape[1] - size[1]) // step + 1):
                positions.append([i * step, j * step, k * stepz])
            positions.append([i * step, images.shape[1] - size[1], k * stepz])
        for j in range((images.shape[1] - size[1]) // step + 1):
            positions.append([images.shape[0] - size[0], j * step, k * stepz])
        positions.append(
            [images.shape[0] - size[0], images.shape[1] - size[1], k * stepz])
    for i in range((images.shape[0] - size[0]) // step + 1):
        for j in range((images.shape[1] - size[1]) // step + 1):
            positions.append([i * step, j * step, images.shape[-1] - size[-1]])
        positions.append(
            [i * step, images.shape[1] - size[1], images.shape[-1] - size[-1]])
    for j in range((images.shape[1] - size[1]) // step + 1):
        positions.append(
            [images.shape[0] - size[0], j * step, images.shape[-1] - size[-1]])
    positions.append([
        images.shape[0] - size[0], images.shape[1] - size[1],
        images.shape[-1] - size[-1]
    ])
    for position in positions:
        img = normalize(images[position[0]:position[0] + size[0],
                               position[1]:position[1] + size[1],
                               position[2]:position[2] + size[2]][np.newaxis,
                                                                  ...])

        img = torch.from_numpy(img).unsqueeze(0).cuda()
        if output == 'multi':
            pre = model(img)[0][0].cpu().numpy() > 0.5
        else:
            pre = model(img)[0].cpu().numpy() > 0.5

        masks[:, position[0]:position[0] + size[0],
              position[1]:position[1] + size[1],
              position[2]:position[2] + size[2]] += pre
    return masks > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--arch', type=str, default='dmmtnet')
    parser.add_argument('--dataset', type=str, default='mosmed')

    parse_config = parser.parse_args()
    print(parse_config)

    if parse_config.dataset == 'mosmed':
        parse_config.data_path = '/raid/wjc/data/mosmed'
    elif parse_config.dataset == 'p20':
        parse_config.data_path = '/raid/wjc/data/covid-19-p20/'

    os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu

    exp_name = parse_config.dataset + '/' + parse_config.arch + '/overall'

    parse_config.txt_dir = './logs/' + exp_name + '/txts'
    parse_config.img_dir = './logs/' + exp_name + '/imgs'

    os.makedirs(parse_config.img_dir, exist_ok=True)
    os.makedirs(parse_config.txt_dir, exist_ok=True)
    logger = create_logger(0, save_dir=parse_config.txt_dir)
    print = logger.info
    print(parse_config)

    cube_size = (160, 160, 32)

    if parse_config.arch == 'vnet':
        from torch_network.vnet3d import VNet
        model = VNet().cuda()
        log_dir = 'logs/mosmed/vnet'
        output = 'single'
    elif parse_config.arch == 'unet':
        from torch_network.unet3d import UNet3D
        model = UNet3D(1).cuda()
        log_dir = 'logs/mosmed/unet'
        output = 'single'
    elif parse_config.arch == 'dmmtnet':
        from torch_network.dmmt import DMMTNet
        model = DMMTNet(1, outputs='multi').cuda()
        log_dir = 'logs/mosmed/dmmtnet_multi_mt_0.1'
        output = 'multi'
    else:
        raise NotImplementedError

    print("Compiled...")
    best_weight = 0

    from utils.SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance

    def nsd(pre, label):
        surface_distances = compute_surface_distances(label, pre, [1, 1, 8])
        v = compute_surface_dice_at_tolerance(surface_distances, 8)
        return v

    eval_funcs = [md.dc, md.jc, nsd, md.asd, md.hd95]
    eval_names = ['dice', 'jaccard', 'nsd', 'asd', 'hd95']
    scores = []
    print('[INFO] Start testing...')

    for fold in range(5):
        train_ids, test_ids = load_division(fold)
        test_images = [
            os.path.join(parse_config.data_path, 'processed/labeled/', test_id,
                         'image.nii') for test_id in test_ids
        ]
        test_labels = [
            os.path.join(parse_config.data_path, 'processed/labeled/', test_id,
                         'label.nii') for test_id in test_ids
        ]
        tic = 0

        model_dir = os.path.join(log_dir,
                                 'fold_{}/models/best_model.t7'.format(fold))
        model.load_state_dict(torch.load(model_dir))

        model.eval()
        dices = []

        for i in range(len(test_images)):
            images, _, _, _ = load_image(test_images[i])
            labels, _, _, _ = load_image(test_labels[i])
            with torch.no_grad():
                prediction = predict_sample(images,
                                            model,
                                            cube_size,
                                            output=output)
            score = []
            context = test_images[i].split('/')[-2]
            for i in range(len(eval_funcs)):
                eval_func = eval_funcs[i]
                v = eval_func(prediction[0], labels > 0.5)
                if i <= 2:
                    v *= 100
                score.append(v)
                context += ' {}:{:.2f}'.format(eval_names[i], score[i])
            print(context)
            scores.append(score)

    scores = np.array(scores)
    avg_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)

    context = ''
    for i in range(len(eval_funcs)):
        context += '&${:.2f}\pm{:.2f}$'.format(avg_scores[i], std_scores[i])
    context += '\\\\'
    print(context)