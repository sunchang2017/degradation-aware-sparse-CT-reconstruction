import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from functools import partial


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, groups=1, act_func=nn.ReLU()):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.groups = groups
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 1, padding=0, groups=self.groups)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1,groups=self.groups)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)
        return out


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu', nn.ReLU(inplace=True)),
        self.add_module('conv', nn.Conv2d(num_input_features, bn_size *
                                          growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        out = torch.cat([x, new_features], 1)
        return out


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        # num_layers = 5
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))


class BlockA(nn.Module):
    def __init__(self, growth_rate=16, block_config=4,
                 num_features=16, bn_size=4, drop_rate=0):
        super(BlockA, self).__init__()
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.block = _DenseBlock(num_layers=block_config,
                                 num_input_features=num_features,
                                 bn_size=bn_size,
                                 growth_rate=growth_rate,
                                 drop_rate=drop_rate)
        self.trans = _Transition((block_config + 1) * num_features, num_features)

    def forward(self, x):
        x1, indices = self.maxpool(x)
        x2 = self.block(x1)
        x3 = self.trans(x2)
        return x3, indices



class SE_Block(nn.Module):
    def __init__(self, ch_in=64, reduction=2):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class look_SE_Block(nn.Module):
    #此模型和SE_Block相同，仅为了观察中间变量
    def __init__(self, ch_in=64, reduction=2):
        super(look_SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x), y


class DCT_SE_DDNet3(nn.Module):
    def __init__(self, growth_rate=32, block_config=4, do_SE=True, do_DCT=True,
                 num_features=32, bn_size=4, drop_rate=0, num_classes=1000, reduction_forward=2):
        super(DCT_SE_DDNet3, self).__init__()

        self.do_SE = do_SE
        self.do_DCT = do_DCT
        # first convolution
        self.dctConv = nn.Conv2d(1, 64, 8, 8, bias=False)  # 1 h w -> 64 h/8 w/8
        if do_SE:
            self.forward_attention_layer = SE_Block(ch_in=64, reduction=reduction_forward)
        self.fistConv = nn.Conv2d(64, num_features, kernel_size=1, stride=1, padding=0, bias=False)

        # A1
        self.A1 = BlockA(growth_rate=growth_rate, block_config=4, num_features=num_features, bn_size=4, drop_rate=0)

        # A2
        self.A2 = BlockA(growth_rate=growth_rate, block_config=4, num_features=num_features, bn_size=4, drop_rate=0)

        # A3
        self.A3 = BlockA(growth_rate=growth_rate, block_config=4, num_features=num_features, bn_size=4, drop_rate=0)

        # B1_Unpooling
        self.AB_Unpool = nn.MaxUnpool2d(kernel_size=2, padding=0, stride=2)

        # B1 + B2 Unpooling
        self.B1_B2Unpool = nn.Sequential(OrderedDict([
            ('B1_deconv1', nn.ConvTranspose2d(in_channels=2 * num_features,
                                              out_channels=2 * num_features,
                                              kernel_size=5, padding=2)),
            ('B1_ReLU1', nn.ReLU(inplace=True)),
            ('B1_BN1', nn.BatchNorm2d(2 * num_features)),
            ('B1_deconv2', nn.ConvTranspose2d(in_channels=2 * num_features,
                                              out_channels=num_features,
                                              kernel_size=1)),
            ('B1_ReLU2', nn.ReLU(inplace=True)),
            ('B1_BN2', nn.BatchNorm2d(num_features)),
        ]))
        # ('B2_MaxUnPooling', nn.MaxUnpool2d(kernel_size=2))
        self.B2_MaxUnPooling = nn.MaxUnpool2d(kernel_size=2, padding=0, stride=2)

        # B2 + B3 Unpooling
        self.B2_B3Unpool = nn.Sequential(OrderedDict([
            ('B2_deconv1', nn.ConvTranspose2d(in_channels=2 * num_features,
                                              out_channels=2 * num_features,
                                              kernel_size=5, padding=2)),
            ('B2_ReLU1', nn.ReLU(inplace=True)),
            ('B2_BN1', nn.BatchNorm2d(2 * num_features)),
            ('B2_deconv2', nn.ConvTranspose2d(in_channels=2 * num_features,
                                              out_channels=num_features,
                                              kernel_size=1)),
            ('B2_ReLU2', nn.ReLU(inplace=True)),
            ('B2_BN2', nn.BatchNorm2d(num_features)),

        ]))
        # ('B2_MaxUnPooling', nn.MaxUnpool2d(kernel_size=2))
        self.B3_MaxUnPooling = nn.MaxUnpool2d(kernel_size=2, padding=0, stride=2)

        # B3-Finish
        self.B3 = nn.Sequential(OrderedDict([
            ('B4_deconv1', nn.ConvTranspose2d(in_channels=2 * num_features,
                                              out_channels=2 * num_features,
                                              kernel_size=5, padding=2)),
            ('B4_ReLU1', nn.ReLU(inplace=True)),
            ('B4_deconv2', nn.ConvTranspose2d(in_channels=2 * num_features,
                                              out_channels=num_features,
                                              kernel_size=1)),
            ('B4_ReLU2', nn.ReLU(inplace=True))
        ]))

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

            elif isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

        self.LastConv = nn.Conv2d(num_features, 64, kernel_size=1, stride=1, padding=0, bias=False) # 1 h w -> 64 h/8 w/8
        self.dct_inconv =  nn.ConvTranspose2d(64, 1, kernel_size=8, stride=8, padding=0, bias=None)

        if self.do_DCT:
            self.weight = torch.from_numpy(np.load('Bz.npy')).float().squeeze().permute(2, 0, 1).unsqueeze(
                1)  # 64 1 8 8, order in Z
            self.dctConv.weight.data = self.weight  # 64 1 8 8
            self.dct_inconv.weight.data = self.weight


    def forward(self, x):
        x_dct = self.dctConv(x)

        if self.do_SE:
            x_dct = self.forward_attention_layer(x_dct)
        features = self.fistConv(x_dct)
        # print(features.shape)

        A1, indices1 = self.A1(features)
        # print('A1', A1.shape)

        A2, indices2 = self.A2(A1)
        # print('A2', A2.shape)

        A3, indices3 = self.A3(A2)
        # print('A3', A3.shape)

        AB = self.AB_Unpool(A3, indices3)
        # print('AB', AB.shape)

        # Concatenation 1
        cat = torch.cat([AB, A2], 1)
        # print('cat1',cat.shape)
        cache = self.B1_B2Unpool(cat)
        # print(cache.shape)
        cache = self.B2_MaxUnPooling(cache, indices2)
        # print(cache.shape)

        # Concatenation 2
        cat = torch.cat([cache, A1], 1)
        # print('cat2', cat.shape)
        cache = self.B2_B3Unpool(cat)
        # print(cache.shape)
        cache = self.B3_MaxUnPooling(cache, indices1)
        # print(cache.shape)

        # Concatenation 3
        cat = torch.cat([cache, features], 1)
        # print('cat3', cat.shape)

        out = self.B3(cat)
        # print('out', out.shape)

        out = self.LastConv(out)
        out = self.dct_inconv(out)
        return out


class Predict_Construction_Error(nn.Module):
    def __init__(self, in_channels=2, nb_filter=(16, 32, 64, 128), groups=1,
                 device='cuda'):
        super().__init__()
        self.groups = groups
        self.device = device

        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)
        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0], groups=self.groups)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], groups=self.groups)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], groups=self.groups)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], groups=self.groups)

        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2], groups=self.groups)
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1], groups=self.groups)
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0], groups=self.groups)

        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1, groups=self.groups)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_1)], 1))

        output = self.final(x0_1)
        return output



class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class RefineModel(nn.Module):
    def __init__(self, in_channels, out_channels, n_resblocks, n_feats, conv=default_conv):
        super(RefineModel, self).__init__()
        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        m_head = [conv(in_channels, n_feats, kernel_size)]

        # define body module
        m_body = [ResBlock(conv, n_feats, kernel_size, act=act) for _ in range(n_resblocks)]
        m_tail = [conv(n_feats, out_channels, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

