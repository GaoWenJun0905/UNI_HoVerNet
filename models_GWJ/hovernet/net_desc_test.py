import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# 20260204_GWJ: 手动实现 UNI (ViT-Large/16) 内部组件，不依赖外部调包
# 调整命名以匹配 timm/UNI 官方 Key (qkv bias, proj weight等)
class UNI_Attention(nn.Module):
    def __init__(self, dim, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class UNI_MLP(nn.Module):
    """适配官方 ViT 的 MLP 命名习惯 (fc1, fc2)"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class UNI_Block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = UNI_Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = UNI_MLP(dim, dim * 4)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class UNI_PatchEmbed(nn.Module):
    """适配官方 PatchEmbed 命名 (proj)"""
    def __init__(self, img_size=224, patch_size=16, embed_dim=1024):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x


class UNIVisionTransformer(nn.Module):
    """手动实现 UNI 骨干网络 (ViT-L/16)，键名适配官方 timm 权重"""

    def __init__(self, img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16):
        super().__init__()
        self.patch_embed = UNI_PatchEmbed(img_size, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.blocks = nn.ModuleList([UNI_Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def get_intermediate_layers(self, x, n=(7, 15, 23, 23)):
        x = self.patch_embed(x)  # [B, 1024, 14, 14]
        x = x.flatten(2).transpose(1, 2)  # [B, 196, 1024]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in n:
                # 20260204_GWJ: 标准做法，提取特征后过一次 Norm
                outputs.append(self.norm(x))
        return outputs


####
# 20260204_GWJ: 定义 UNI 适配器 (使用反卷积并包含线性投影层)
class UNI_Adapter(nn.Module):
    """
    将 UNI (ViT-L) 的 14x14 特征图转换为 ResNet 风格的金字塔特征 (d0, d1, d2, d3)。
    在每个分支头部加入 1x1 卷积作为线性投影层，负责特征提炼。
    使用转置卷积 (Deconvolution) 进行上采样。
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
        # feats 是一个列表，包含 UNI 不同层的输出 [feat0, feat1, feat2, feat3]

        # 20260204_GWJ: 先投影提炼特征，再进行空间变换（上/下采样）
        d0 = self.up_d0(self.proj_d0(feats[0]))
        d1 = self.up_d1(self.proj_d1(feats[1]))
        d2 = self.up_d2(self.proj_d2(feats[2]))
        d3 = self.down_d3(self.proj_d3(feats[3]))

        return d0, d1, d2, d3


####
class HoVerNet(Net):
    """Initialise HoVer-Net."""

    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        assert mode == 'original' or mode == 'fast', \
            'Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`.' % mode

        # --- 20260204_GWJ: 使用手动实现的 UNI (ViT-L) 替换原生 ResNet ---
        self.backbone = UNIVisionTransformer(
            img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16
        )

        # 实例化反卷积适配器
        self.adapter = UNI_Adapter(embed_dim=1024)
        # --------------------------------------------------

        # 这里的 conv_bot 保留，用于处理 d3 (最深层特征)
        # Adapter 输出的 d3 已经是 2048 通道，这里将其降维到 1024 供 Decoder 使用
        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

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

            # 20260204_GWJ: 如果是分类头，在 u1 这一级加入 SEModule
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

        self.upsample2x = UpSample2x()
        # TODO: pytorch still require the channel eventhough its ignored
        self.weights_init()

    def forward(self, imgs):

        imgs = imgs / 255.0  # to 0-1 range to match XY

        # --- 20260204_GWJ: UNI Encoder Forward 逻辑 ---
        if imgs.shape[2:] != (224, 224):
            imgs_input = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
        else:
            imgs_input = imgs

        with torch.set_grad_enabled(not self.freeze):
            # 获取 Block 7, 15, 23, 23 的输出
            feats_tokens = self.backbone.get_intermediate_layers(imgs_input, n=(7, 15, 23, 23))

            # Helper: Token -> 2D Feature Map
            def token_to_2d(x):
                # x shape: [B, 197, 1024] -> [B, 1024, 14, 14]
                B, N, C = x.shape
                h = w = int((N - 1) ** 0.5)  # 14
                # 去掉 CLS token (index 0)
                return x[:, 1:, :].transpose(1, 2).reshape(B, C, h, w)

            feats_2d = [token_to_2d(f) for f in feats_tokens]

            # 通过 Adapter 生成金字塔特征 (含线性投影层)
            d0, d1, d2, d3 = self.adapter(feats_2d)

        # 这里的 d3 是 Adapter 输出的 2048 通道，conv_bot 把它变成 1024
        d3 = self.conv_bot(d3)
        d = [d0, d1, d2, d3]
        # ----------------------------------------------

        # TODO: switch to `crop_to_shape` ?
        if self.mode == 'original':
            d[0] = crop_op(d[0], [184, 184])
            d[1] = crop_op(d[1], [72, 72])
        else:
            d[0] = crop_op(d[0], [92, 92])
            d[1] = crop_op(d[1], [36, 36])

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
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