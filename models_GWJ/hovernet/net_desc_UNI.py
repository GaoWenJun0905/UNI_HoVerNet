import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .net_utils import (DenseBlock, Net, ResidualBlock, TFSamepaddingLayer,
                        UpSample2x)
from .utils import crop_op, crop_to_shape


####
# 20260204_GWJ: 定义 SE 注意力模块
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


####
# 20260204_GWJ: 定义 UNI 适配器 (针对 ViT-L/16 设计)
class UNI_Adapter(nn.Module):
    """
    将 UNI (ViT-L/16) 的 14x14 特征图转换为 ResNet 风格的金字塔特征 (d0, d1, d2, d3)。
    在每个分支头部加入 1x1 卷积作为线性投影层，负责特征提炼。
    """

    def __init__(self, embed_dim=1024):
        super().__init__()

        # d0: 目标尺寸 56x56 (1/4), 通道 256
        self.proj_d0 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.up_d0 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),  # 14->28
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 28->56
        )

        # d1: 目标尺寸 28x28 (1/8), 通道 512
        self.proj_d1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.up_d1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),  # 14->28
        )

        # d2: 目标尺寸 14x14 (1/16), 通道 1024
        self.proj_d2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.up_d2 = nn.Sequential(
            nn.Conv2d(embed_dim, 1024, kernel_size=1, stride=1),  # 14->14
        )

        # d3: 目标尺寸 7x7 (1/32), 通道 2048
        self.proj_d3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.down_d3 = nn.Sequential(
            nn.Conv2d(embed_dim, 2048, kernel_size=3, stride=2, padding=1),  # 14->7
        )

    def forward(self, feats):
        # feats 是从 timm 提取的中间层输出列表 [feat0, feat1, feat2, feat3]
        d0 = self.up_d0(self.proj_d0(feats[0]))
        d1 = self.up_d1(self.proj_d1(feats[1]))
        d2 = self.up_d2(self.proj_d2(feats[2]))
        d3 = self.down_d3(self.proj_d3(feats[3]))
        return d0, d1, d2, d3


####
class HoVerNet(Net):
    """Initialise HoVer-Net with UNI (ViT-L/16) Backbone."""

    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        assert mode == 'original' or mode == 'fast', \
            'Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`.' % mode

        # --- 20260204_GWJ: 定义并行子模块，确保命名路径处于顶级 ---
        self.backbone = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            img_size=224,
            num_classes=0,
            dynamic_img_size=True
        )

        self.adapter = UNI_Adapter(embed_dim=1024)

        # 这里的 conv_bot 直接定义在 self 下，确保加载时 key 为 "conv_bot.weight"
        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        # 这里的 upsample2x 直接定义在 self 下，确保加载时 key 包含 "upsample2x.unpool_mat"
        self.upsample2x = UpSample2x()
        # --------------------------------------------------

        # 修改 create_decoder_branch 以支持 SE Attention
        def create_decoder_branch(out_ch=2, ksize=5, is_tp=False):
            module_list = [
                ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            # 20260204_GWJ: 在分类头加入 SEModule
            module_list_u1 = [
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False),),
            ]
            if is_tp:
                module_list_u1.append(("se", SEModule(64)))  # 注入 SE 注意力
            u1 = nn.Sequential(OrderedDict(module_list_u1))

            module_list = [
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0), ])
            )
            return decoder

        ksize = 5 if mode == 'original' else 3
        if nr_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )
        else:
            # 20260119_GWJ 修改的双分类头结构，并在 20260204 注入 SE
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp_TTF-1", create_decoder_branch(ksize=ksize, out_ch=nr_types, is_tp=True)),
                        ("tp_WT-1", create_decoder_branch(ksize=ksize, out_ch=nr_types, is_tp=True)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )

        self.weights_init()

    def forward(self, imgs):

        imgs = imgs / 255.0  # to 0-1 range

        # --- 20260204_GWJ: UNI Encoder Forward 逻辑 ---
        if imgs.shape[2:] != (224, 224):
            imgs_input = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
        else:
            imgs_input = imgs

        with torch.set_grad_enabled(not self.freeze):
            # 获取中间层特征
            indices = [7, 15, 23, 23]
            feats_2d = self.backbone.get_intermediate_layers(imgs_input, n=indices, reshape=True)

            # 通过 Adapter 生成金字塔特征
            d0, d1, d2, d3 = self.adapter(feats_2d)

        # 核心：使用顶级定义的 conv_bot 处理最深层特征
        d3 = self.conv_bot(d3)
        d = [d0, d1, d2, d3]
        # ----------------------------------------------

        # 裁剪操作
        if self.mode == 'original':
            d[0] = crop_op(d[0], [184, 184])
            d[1] = crop_op(d[1], [72, 72])
        else:
            d[0] = crop_op(d[0], [92, 92])
            d[1] = crop_op(d[1], [36, 36])

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            # 使用顶级定义的 upsample2x
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict


####
def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        assert "Unknown Model Mode %s" % mode
    return HoVerNet(mode=mode, **kwargs)