import torch
import torch.nn as nn
import torch.nn.functional as F


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(
                input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(input, self.running_mean, self.running_var,
                            self.weight, self.bias, True, self.momentum,
                            self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


class Se_Block(nn.Module):
    def __init__(self, in_chan):
        super(Se_Block, self).__init__()
        self.avg = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear1 = nn.Linear(in_chan, in_chan // 8)
        self.relu1 = nn.ReLU(in_chan // 8)
        self.linear2 = nn.Linear(in_chan // 8, in_chan)

    def forward(self, x, y):
        b, c, _, _, _ = y.size()
        y = self.avg(y).view(b, c)
        y = self.linear1(y)
        y = self.relu1(y)
        y = self.linear2(y)
        y = F.sigmoid(y)
        x = x + x * y[:, :, None, None, None]
        return x


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    base = 16
    if double_chnnel:
        layer1 = LUConv(in_channel, base * (2**(depth + 1)), act)
        layer2 = LUConv(base * (2**(depth + 1)), base * (2**(depth + 1)), act)
    else:
        layer1 = LUConv(in_channel, base * (2**depth), act)
        layer2 = LUConv(base * (2**depth), base * (2**depth) * 2, act)

    return nn.Sequential(layer1, layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channel, depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth, act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth, act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans,
                                          outChans,
                                          kernel_size=2,
                                          stride=2)
        self.ops = _make_nConv(inChans + outChans // 2,
                               depth,
                               act,
                               double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv, skip_x), 1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)

    def forward(self, x):
        out = self.final_conv(x)
        return out


class DMMTNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=1, act='relu', outputs='single'):
        super(DMMTNet, self).__init__()

        base_filter = 32

        self.down_tr64 = DownTransition(1, 0, act)
        self.down_tr128 = DownTransition(base_filter, 1, act)
        self.down_tr256 = DownTransition(base_filter * 2, 2, act)
        self.down_tr512 = DownTransition(base_filter * 4, 3, act)

        self.up_tr256 = UpTransition(base_filter * 8, base_filter * 8, 2, act)
        self.up_tr128 = UpTransition(base_filter * 4, base_filter * 4, 1, act)
        self.up_tr64 = UpTransition(base_filter * 2, base_filter * 2, 0, act)

        self.attention_block_all_levels = []
        for _ in range(3):
            self.attention_blocks = []
            for level in [1, 2, 4, 8]:
                self.attention_blocks.append(
                    Se_Block(base_filter * level).cuda())
            self.attention_block_all_levels.append(self.attention_blocks)

        self.outputs = outputs
        if outputs == 'single':
            self.out_tr = OutputTransition(base_filter, n_class)
        else:
            self.out_tr_x1 = OutputTransition(base_filter, n_class)
            self.out_tr_x2 = OutputTransition(base_filter * 2, n_class)
            self.out_tr_x4 = OutputTransition(base_filter * 4, n_class)
            self.out_tr_x8 = OutputTransition(base_filter * 8, n_class)

    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128, self.skip_out128 = self.down_tr128(self.out64)
        self.out256, self.skip_out256 = self.down_tr256(self.out128)
        self.out512, self.skip_out512 = self.down_tr512(self.out256)

        x1 = F.max_pool3d(x, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        x2 = F.max_pool3d(x, kernel_size=(1, 2, 1), stride=(1, 2, 1))
        x3 = F.max_pool3d(x, kernel_size=(1, 1, 2), stride=(1, 1, 2))

        for i, _tmp in enumerate([x1, x2, x3]):
            attention_blocks = self.attention_block_all_levels[i]
            _, f_level_1 = self.down_tr64(_tmp)
            _, f_level_2 = self.down_tr128(self.out64)
            _, f_level_3 = self.down_tr256(self.out128)
            _, f_level_4 = self.down_tr512(self.out256)
            self.skip_out64 = attention_blocks[0](self.skip_out64, f_level_1)
            self.skip_out128 = attention_blocks[1](self.skip_out128, f_level_2)
            self.skip_out256 = attention_blocks[2](self.skip_out256, f_level_3)
            self.skip_out512 = attention_blocks[3](self.skip_out512, f_level_4)

        self.out512 = self.skip_out512

        self.out_up_256 = self.up_tr256(self.out512, self.skip_out256)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)

        if self.outputs == 'single':
            self.out = self.out_tr(self.out_up_64)
        elif self.outputs == 'multi':
            self.out = []
            self.out.append(self.out_tr_x1(self.out_up_64))
            self.out.append(self.out_tr_x2(self.out_up_128))
            self.out.append(self.out_tr_x4(self.out_up_256))
            self.out.append(self.out_tr_x8(self.out512))
        else:
            raise NotImplementedError

        return self.out