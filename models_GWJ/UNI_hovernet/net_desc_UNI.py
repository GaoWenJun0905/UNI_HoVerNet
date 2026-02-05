# -*- coding: utf-8 -*-
import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .net_utils import (DenseBlock, Net, TFSamepaddingLayer, UpSample2x)
from .utils import crop_to_shape


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
# 20260204_GWJ: 参考 CellViT 深度定制的 UNI 适配器
class UNI_Adapter(nn.Module):
    """
    核心思路：将 ViT 恒定的 14x14 特征图，通过转置卷积 (Deconv)
    构建成 80x80, 40x40, 20x20, 10x10 的金字塔。
    """

    def __init__(self, embed_dim=1024):
        super().__init__()
        # 对应 ViT 的 4 个阶段输出的归一化
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(4)])

        # d0: 1/4 尺度 (目标 80x80)。从 14x14 经过 4x4 stride=4 卷积得到 56x56，再补到 80
        self.up_d0 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(80, 80), mode='bilinear', align_corners=False)
        )

        # d1: 1/8 尺度 (目标 40x40)。14x14 -> 28x28 -> 40x40
        self.up_d1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(40, 40), mode='bilinear', align_corners=False)
        )

        # d2: 1/16 尺度 (目标 20x20)。14x14 -> 20x20
        self.up_d2 = nn.Sequential(
            nn.Upsample(size=(20, 20), mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # d3: 1/32 尺度 (目标 10x10)。14x14 -> 10x10
        self.down_d3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((10, 10)),
            nn.Conv2d(embed_dim, 2048, kernel_size=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats):
        processed_feats = []
        for i, f in enumerate(feats):
            # 去掉 ViT 的 Class Token (B, 197, 1024 -> B, 196, 1024)
            if f.shape[1] == 197: f = f[:, 1:, :]
            b, l, c = f.shape
            grid_size = int(math.sqrt(l))
            # 还原为 2D 形状
            f = f.transpose(-1, -2).view(b, c, grid_size, grid_size).contiguous()
            # 归一化 (CellViT 稳定数值的关键)
            f = f.permute(0, 2, 3, 1)
            f = self.norms[i](f)
            f = f.permute(0, 3, 1, 2).contiguous()
            processed_feats.append(f)

        return (self.up_d0(processed_feats[0]),
                self.up_d1(processed_feats[1]),
                self.up_d2(processed_feats[2]),
                self.down_d3(processed_feats[3]))


####
class HoVerNet(Net):
    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types

        # 加载 UNI 骨干 (ViT-L/16)
        self.backbone = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            img_size=224,
            num_classes=0,
            dynamic_img_size=True,
            init_values=1e-5
        )

        self.adapter = UNI_Adapter(embed_dim=1024)
        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5, is_tp=False):
            # 解码器 Block 保持 HoVerNet 原生结构，但输入尺寸已经过 Adapter 扩容
            u3 = nn.Sequential(OrderedDict([
                ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False))
            ]))
            u2 = nn.Sequential(OrderedDict([
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False))
            ]))
            u1_list = [
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False)),
            ]
            if is_tp: u1_list.append(("se", SEModule(64)))
            u1 = nn.Sequential(OrderedDict(u1_list))

            u0 = nn.Sequential(OrderedDict([
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True))
            ]))
            return nn.ModuleList([u3, u2, u1, u0])

        ksize = 5 if mode == 'original' else 3
        branch_configs = ["np", "hv"] + (["tp_TTF-1", "tp_WT-1"] if nr_types else [])
        self.decoder = nn.ModuleDict({
            name: create_decoder_branch(ksize=ksize, out_ch=nr_types if "tp" in name else 2, is_tp="tp" in name)
            for name in branch_configs
        })

        self.upsample2x = UpSample2x()
        self.weights_init()

    def forward(self, imgs):
        imgs = imgs / 255.0
        # 记录原始输入的 H, W，或者直接设定目标尺寸 164 (HoVerNet 默认)
        # target_size = (164, 164)

        # 强制缩放至 224 给 UNI 编码
        imgs_input = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False) if imgs.shape[2:] != (
        224, 224) else imgs

        with torch.set_grad_enabled(not self.freeze):
            indices = [5, 11, 17, 23]
            feats = self.backbone.get_intermediate_layers(imgs_input, n=indices)
            d_list = list(self.adapter(feats))

        d_list[3] = self.conv_bot(d_list[3])
        out_dict = OrderedDict()

        for branch_name, branch_modules in self.decoder.items():
            # u3: 10->20
            u3 = self.upsample2x(d_list[3])
            u3 = F.interpolate(u3, size=d_list[2].shape[2:], mode='bilinear', align_corners=False)
            u3 = branch_modules[0](u3 + d_list[2])

            # u2: 20->40
            u2 = self.upsample2x(u3)
            u2 = F.interpolate(u2, size=d_list[1].shape[2:], mode='bilinear', align_corners=False)
            u2 = branch_modules[1](u2 + d_list[1])

            # u1: 40->80
            u1 = self.upsample2x(u2)
            u1 = F.interpolate(u1, size=d_list[0].shape[2:], mode='bilinear', align_corners=False)
            u1 = branch_modules[2](u1 + d_list[0])

            # u0: 得到基础预测 (此时是 80x80)
            u0 = branch_modules[3](u1)

            # --- 关键修复：将 80x80 放大到 164x164 以匹配 Label ---
            # 注意：HoVerNet 的输出尺寸通常取决于配置，164 是你报错信息里提示的 true 的尺寸
            u0 = F.interpolate(u0, size=(164, 164), mode='bilinear', align_corners=False)

            out_dict[branch_name] = u0

        return out_dict


####
def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        assert "Unknown Model Mode %s" % mode
    return HoVerNet(mode=mode, **kwargs)