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
from test import predict_sample

import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--arch', type=str)
parser.add_argument('--dataset', type=str, default='mosmed')
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--fold', type=str)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--bt_size', type=int, default=8)
parser.add_argument('--aug', type=int, default=1)
parser.add_argument('--at_func', type=str, default='channel')
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--ver', type=str)

parse_config = parser.parse_args()
print(parse_config)

if parse_config.dataset == 'mosmed':
    parse_config.data_path = '/raid/wjc/data/mosmed'

os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu

exp_name = parse_config.dataset + '/' + parse_config.arch + '/fold_' + str(
    parse_config.fold)

parse_config.log_dir = 'logs/' + exp_name + '/logs'
parse_config.txt_dir = 'logs/' + exp_name + '/txts'
parse_config.model_dir = 'logs/' + exp_name + '/models'

os.makedirs(parse_config.log_dir, exist_ok=True)
os.makedirs(parse_config.txt_dir, exist_ok=True)
os.makedirs(parse_config.model_dir, exist_ok=True)
logger = create_logger(0, save_dir=parse_config.txt_dir)
summary_writer = SummaryWriter(parse_config.log_dir)
print = logger.info
print(parse_config)

cube_size = (160, 160, 32)

if parse_config.arch == 'vnet':
    from torch_network.vnet3d import VNet
    model = VNet().cuda()
    parse_config.output = 'single'
elif parse_config.arch == 'unet':
    from torch_network.unet3d import UNet3D
    model = UNet3D(1).cuda()
    parse_config.output = 'single'
elif parse_config.arch == 'nnunet':
    from torch_network.nn_unet3d import nnUNet3D
    model = nnUNet3D(1).cuda()
    parse_config.output = 'single'
elif 'dmmtnet' in parse_config.arch:
    from torch_network.dmmt import DMMTNet
    if 'multi' in parse_config.arch:
        model = DMMTNet(1, outputs='multi').cuda()
        parse_config.output = 'multi'
        print('[INFO] Output: ' + parse_config.output)
    else:
        model = DMMTNet(1, outputs='single').cuda()
        parse_config.output = 'single'
    if parse_config.pretrain:
        pre_path = os.path.join(parse_config.model_dir,
                                'best_model.t7').replace(
                                    parse_config.arch, 'unet')
        model.load_state_dict(torch.load(pre_path, map_location='cuda:0'),
                              strict=False)
else:
    raise NotImplementedError

optimizer = torch.optim.Adam(model.parameters(), lr=parse_config.lr)
criteon = dice_loss
score = [dice_score]

print("Compiled...")
best_weight = 0

train_ids, test_ids = load_division(parse_config.fold)
train_images = [
    os.path.join(parse_config.data_path, 'processed/labeled/', train_id,
                 'image.nii') for train_id in train_ids
]
train_labels = [
    os.path.join(parse_config.data_path, 'processed/labeled/', train_id,
                 'label.nii') for train_id in train_ids
]
test_images = [
    os.path.join(parse_config.data_path, 'processed/labeled/', test_id,
                 'image.nii') for test_id in test_ids
]
test_labels = [
    os.path.join(parse_config.data_path, 'processed/labeled/', test_id,
                 'label.nii') for test_id in test_ids
]
print('[INFO] Train on {} cases and test on {} cases'.format(
    len(train_images), len(test_images)))

dataset = mosmed(train_images,
                 train_labels,
                 sample_size=cube_size,
                 output=parse_config.output,
                 bs=parse_config.bt_size)
train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=parse_config.bt_size,
                                           num_workers=2,
                                           pin_memory=True)
tic = 0
for k in range(parse_config.n_epochs):
    model.train()
    tic = time.perf_counter()
    for batch_idx, batch_data in enumerate(train_loader):
        images = batch_data[0].cuda().float().unsqueeze(1)
        labels = batch_data[1].cuda().float().unsqueeze(1)

        pre = model(images)
        if parse_config.output == 'single':
            pre = F.sigmoid(pre)
            loss = criteon(pre, labels)
        elif parse_config.output == 'multi':
            loss = 0
            for level, _pre in enumerate(pre):
                _label = F.max_pool3d(labels, (2**level))
                _pre = F.sigmoid(_pre)
                assert _pre.shape == _label.shape
                loss += criteon(_pre, _label)
        else:
            raise NotImplementedError

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            duration = time.perf_counter() - tic
            tic = time.perf_counter()
            print('[%d/%d-%d/%d]' %
                  (k, parse_config.n_epochs, batch_idx, len(train_loader)) +
                  'loss:{:.4f}  Time:{:.4f}'.format(loss.item(), duration))

            summary_writer.add_scalar(
                'Fold-{}/SegLoss'.format(parse_config.fold), loss,
                k * len(train_loader) + batch_idx)

    model.eval()
    dices = []
    print('[INFO] Start testing...')
    for i in range(len(test_images)):
        images, _, _, _ = load_image(test_images[i])
        labels, _, _, _ = load_image(test_labels[i])
        with torch.no_grad():
            prediction = predict_sample(images, model, cube_size,
                                        parse_config.output)
        dices.append(dsc(prediction[0], labels > 0.5))
    print('[Test] Avg Dice:{:.2f}'.format(np.mean(dices)))
    summary_writer.add_scalar('Fold-{}/TestDSC'.format(parse_config.fold),
                              np.mean(dices), k)

    if np.mean(dices) > best_weight:
        best_weight = np.mean(dices)
        torch.save(model.state_dict(),
                   os.path.join(parse_config.model_dir, 'best_model.t7'))
    torch.save(model.state_dict(),
               os.path.join(parse_config.model_dir, 'latest_model.t7'))
