import sys, os, argparse
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
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

parser.add_argument('--cons', type=float)

parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--bt_size', type=int, default=8)
parser.add_argument('--aug', type=int, default=1)
parser.add_argument('--at_func', type=str, default='channel')
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--ver', type=str)

parse_config = parser.parse_args()

print(parse_config)

assert parse_config.arch == 'dmmtnet_multi_mt'

if parse_config.dataset == 'mosmed':
    parse_config.data_path = '/raid/wjc/data/mosmed'

os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu

exp_name = parse_config.dataset + '/' + parse_config.arch + '_' + str(
    parse_config.cons) + '/fold_' + str(parse_config.fold)

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

from torch_network.dmmt import DMMTNet

teacher_model = DMMTNet(1, outputs='multi').cuda()
student_model = DMMTNet(1, outputs='multi').cuda()
parse_config.output = 'multi'
print('[INFO] Output: ' + parse_config.output)
if parse_config.pretrain:
    pre_path = os.path.join(parse_config.model_dir, 'best_model.t7').replace(
        parse_config.arch + '_' + str(parse_config.cons), 'nnunet')
    student_model.load_state_dict(torch.load(pre_path, map_location='cuda:0'),
                                  strict=False)
    teacher_model.load_state_dict(torch.load(pre_path, map_location='cuda:0'),
                                  strict=False)

optimizer = torch.optim.Adam(student_model.parameters(), lr=parse_config.lr)
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
unlabel_images = [
    os.path.join(parse_config.data_path, 'processed/unlabeled/', unlabel_id,
                 'image.nii')
    for unlabel_id in os.listdir(
        os.path.join(parse_config.data_path, 'processed/unlabeled/'))
]
print(
    '[INFO] Train on {} cases and test on {} cases, {} Unlabeled cases'.format(
        len(train_images), len(test_images), len(unlabel_images)))

parse_config.bt_size = parse_config.bt_size // 2
dataset = mosmed(train_images,
                 train_labels,
                 sample_size=cube_size,
                 output=parse_config.output,
                 bs=parse_config.bt_size)
train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=parse_config.bt_size,
                                           num_workers=2,
                                           pin_memory=True)

test_dataset = mosmed(test_images,
                      test_labels,
                      sample_size=cube_size,
                      output=parse_config.output,
                      bs=parse_config.bt_size)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=parse_config.bt_size,
                                          num_workers=2,
                                          pin_memory=True)

u_dataset = UnlabelDataset(unlabel_images,
                           sample_size=cube_size,
                           bs=parse_config.bt_size)
u_loader = torch.utils.data.DataLoader(u_dataset,
                                       batch_size=parse_config.bt_size,
                                       num_workers=2,
                                       pin_memory=True)


def consistency_loss(pre, gt):
    _loss = 0
    for _pre, _gt in zip(pre, gt):
        _pre = F.sigmoid(_pre)
        _gt = Variable(F.sigmoid(_gt).detach().data, requires_grad=False)
        _loss += F.mse_loss(_pre, _gt, size_average=True)
    return _loss


def update_ema_variables(teacher, student, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_current_consistency_weight(epoch,
                                   consistency=parse_config.cons,
                                   consistency_rampup=20):
    def sigmoid_rampup(current, rampup_length):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def update_ema_variables(teacher, student, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


tic = 0
for k in range(parse_config.n_epochs):
    student_model.train()
    teacher_model.train()
    tic = time.perf_counter()
    for batch_idx, (batch_data,
                    u_data) in enumerate(zip(train_loader, u_loader)):
        images = batch_data[0].cuda().float().unsqueeze(1)
        labels = batch_data[1].cuda().float().unsqueeze(1)
        u_images = u_data.cuda().float().unsqueeze(1)
        noise = torch.clamp(torch.randn_like(u_images) * 0.1, -0.2, 0.2)
        pre = student_model(images)

        # supervised loss
        sup_loss = 0
        for level, _pre in enumerate(pre):

            _label = F.max_pool3d(labels, (2**level))

            _pre = F.sigmoid(_pre)
            assert _pre.shape == _label.shape
            sup_loss += criteon(_pre, _label)

        # consistency loss

        teacher_o = teacher_model(u_images + noise)
        student_o = student_model(u_images)
        consis_loss = consistency_loss(student_o, teacher_o)

        loss = consis_loss * get_current_consistency_weight(k) + sup_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step = k * len(train_loader) + batch_idx
        update_ema_variables(teacher_model, student_model, 0.999, step)

        if batch_idx % 10 == 0:
            duration = time.perf_counter() - tic
            tic = time.perf_counter()
            print(
                '[%d/%d-%d/%d]' %
                (k, parse_config.n_epochs, batch_idx, len(train_loader)) +
                'loss:{:.4f} super_loss:{:.4f} consis_loss:{:.4f}  Time:{:.4f}'
                .format(loss.item(), sup_loss.item(), consis_loss.item(),
                        duration))

            summary_writer.add_scalar(
                'Fold-{}/SegLoss'.format(parse_config.fold), loss,
                k * len(train_loader) + batch_idx)
            summary_writer.add_scalar(
                'Fold-{}/sup_loss'.format(parse_config.fold), sup_loss,
                k * len(train_loader) + batch_idx)
            summary_writer.add_scalar(
                'Fold-{}/consis_loss'.format(parse_config.fold), consis_loss,
                k * len(train_loader) + batch_idx)

    student_model.eval()
    dices = []
    print('[INFO] Start testing...')
    for i in range(len(test_images)):
        images, _, _, _ = load_image(test_images[i])
        labels, _, _, _ = load_image(test_labels[i])
        with torch.no_grad():
            prediction = predict_sample(images, student_model, cube_size,
                                        parse_config.output)
        dices.append(dsc(prediction[0], labels > 0.5))
    print('[Test] Avg Dice:{:.2f}'.format(np.mean(dices)))
    summary_writer.add_scalar('Fold-{}/TestDSC'.format(parse_config.fold),
                              np.mean(dices), k)

    if np.mean(dices) > best_weight:
        best_weight = np.mean(dices)
        torch.save(student_model.state_dict(),
                   os.path.join(parse_config.model_dir, 'best_model.t7'))
    torch.save(student_model.state_dict(),
               os.path.join(parse_config.model_dir, 'latest_model.t7'))

    test_loss = 0
    for batch_data in test_loader:
        images = batch_data[0].cuda().float().unsqueeze(1)
        labels = batch_data[1].cuda().float().unsqueeze(1)
        with torch.no_grad():
            pre = student_model(images)
            sup_loss = 0
            for level, _pre in enumerate(pre):
                _label = F.max_pool3d(labels, (2**level))
                _pre = F.sigmoid(_pre)
                assert _pre.shape == _label.shape
                sup_loss += criteon(_pre, _label)
            test_loss += sup_loss
    test_loss /= len(test_loader)
    summary_writer.add_scalar('Fold-{}/TestLoss'.format(parse_config.fold),
                              test_loss, k)
