# -*- coding: utf-8 -*-
import math
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .net_utils import (Net, TFSamepaddingLayer, UpSample2x)
from .utils import crop_to_shape


# =========================================================================
# 🌟 核心：重写一个绝对不会缩减尺寸的 SafeDenseBlock
# =========================================================================
class SafeDenseUnit(nn.Module):
    def __init__(self, in_ch, ksize, ch_list):
        super().__init__()
        pad = ksize // 2
        self.preact_bna = nn.Sequential(
            nn.BatchNorm2d(in_ch, eps=1e-5),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(in_ch, ch_list[0], 1, stride=1, padding=0, bias=False)
        self.conv1_bn = nn.BatchNorm2d(ch_list[0], eps=1e-5)
        self.conv2 = nn.Conv2d(ch_list[0], ch_list[1], ksize, stride=1, padding=pad, bias=False)

    def forward(self, x):
        out = self.preact_bna(x)
        out = self.conv1(out)
        out = F.relu(self.conv1_bn(out), inplace=True)
        out = self.conv2(out)
        return torch.cat([x, out], dim=1)


class SafeDenseBlock(nn.Module):
    def __init__(self, in_ch, ksize, ch_list, nr_units):
        super().__init__()
        self.units = nn.ModuleList()
        cur_ch = in_ch
        actual_k = ksize[1] if isinstance(ksize, list) else ksize
        for _ in range(nr_units):
            self.units.append(SafeDenseUnit(cur_ch, actual_k, ch_list))
            cur_ch += ch_list[1]
        self.blk_bna = nn.Sequential(
            nn.BatchNorm2d(cur_ch, eps=1e-5),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for unit in self.units:
            x = unit(x)
        return self.blk_bna(x)


# =========================================================================
# 🌟 ConvNeXt Base 骨干网络 (参数对齐 Base 版本)
# =========================================================================
class Block(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)))

    def forward(self, x):
        input = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = (self.gamma * x).permute(0, 3, 1, 2)
        return input + x


class ConvNeXt_Base(nn.Module):
    # 🌟 修改点 1：Base 版 depths 为 [3, 3, 27, 3], dims 为 [128, 256, 512, 1024]
    def __init__(self, in_chans=3, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        # Stem
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0], 4, 4),
            nn.GroupNorm(1, dims[0])
        ))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                nn.GroupNorm(1, dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], 2, 2)
            ))
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[Block(dim=dims[i]) for _ in range(depths[i])]))

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)
        return outs


# =========================================================================
# 🌟 MTCA 与辅助组件
# =========================================================================
def INF(B, H, W, device):
    return -torch.diag(torch.tensor(float("inf"), device=device).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class Asymmetric_CCAttention(nn.Module):
    def __init__(self, q_dim, kv_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(q_dim, q_dim // 8, 1)
        self.key_conv = nn.Conv2d(kv_dim, q_dim // 8, 1)
        self.value_conv = nn.Conv2d(kv_dim, q_dim, 1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_q, x_kv):
        B, _, H, W = x_q.size()
        q_H = self.query_conv(x_q).permute(0, 3, 1, 2).reshape(B * W, -1, H).permute(0, 2, 1)
        q_W = self.query_conv(x_q).permute(0, 2, 1, 3).reshape(B * H, -1, W).permute(0, 2, 1)
        k = self.key_conv(x_kv);
        k_H = k.permute(0, 3, 1, 2).reshape(B * W, -1, H);
        k_W = k.permute(0, 2, 1, 3).reshape(B * H, -1, W)
        v = self.value_conv(x_kv);
        v_H = v.permute(0, 3, 1, 2).reshape(B * W, -1, H);
        v_W = v.permute(0, 2, 1, 3).reshape(B * H, -1, W)
        e_H = (torch.bmm(q_H, k_H) + INF(B, H, W, x_q.device)).reshape(B, W, H, H).permute(0, 2, 1, 3)
        e_W = torch.bmm(q_W, k_W).reshape(B, H, W, W)
        att = self.softmax(torch.cat([e_H, e_W], 3))
        o_H = torch.bmm(v_H, att[:, :, :, :H].permute(0, 2, 1, 3).reshape(B * W, H, H).permute(0, 2, 1)).reshape(B, W,
                                                                                                                 -1,
                                                                                                                 H).permute(
            0, 2, 3, 1)
        o_W = torch.bmm(v_W, att[:, :, :, H:].reshape(B * H, W, W).permute(0, 2, 1)).reshape(B, H, -1, W).permute(0, 2,
                                                                                                                  1, 3)
        return self.gamma * (o_H + o_W) + x_q


class Directed_MTCA(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.cc_ttf = nn.Sequential(Asymmetric_CCAttention(ch, ch * 2), Asymmetric_CCAttention(ch, ch * 2))
        self.cc_wt = nn.Sequential(Asymmetric_CCAttention(ch, ch * 2), Asymmetric_CCAttention(ch, ch * 2))

    def forward(self, f_np, f_hv, f_ttf, f_wt):
        m = torch.cat([f_np, f_hv], 1).detach()
        return f_np, f_hv, self.cc_ttf[1](self.cc_ttf[0](f_ttf, m), m), self.cc_wt[1](self.cc_wt[0](f_wt, m), m)


class ConvNeXt_Adapter(nn.Module):
    # 🌟 修改点 2：通道数更新为 Base 版的 [128, 256, 512, 1024]
    def __init__(self, in_c=[128, 256, 512, 1024], out_c=[256, 512, 1024, 2048]):
        super().__init__()
        self.adapt = nn.ModuleList([
            nn.Sequential(nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
            for i, o in zip(in_c, out_c)
        ])

    def forward(self, feats): return [m(f) for m, f in zip(self.adapt, feats)]


# =========================================================================
# 🌟 核心 4：主模型实现 (自愈型 Decoder)
# =========================================================================
class ConvNeXt_Base_TripMTCA_HoVerNet(Net):
    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):
        super().__init__()
        self.nr_types = nr_types
        self.freeze = freeze

        # 🌟 修改点 3：实例化 Base 版骨干
        self.backbone = ConvNeXt_Base(in_chans=input_ch)

        if self.freeze:
            for p in self.backbone.parameters(): p.requires_grad = False

        self.adapter = ConvNeXt_Adapter()
        self.conv_bot = nn.Conv2d(2048, 1024, 1, bias=False)

        def create_branch(out_ch=2, ksize=5):
            pad = ksize // 2
            u3 = nn.Sequential(OrderedDict([
                ("conva", nn.Conv2d(1024, 256, ksize, padding=pad, bias=False)),
                ("dense", SafeDenseBlock(256, [1, ksize], [128, 32], 8)),
                ("convf", nn.Conv2d(512, 512, 1, bias=False))
            ]))
            u2 = nn.Sequential(OrderedDict([
                ("conva", nn.Conv2d(512, 128, ksize, padding=pad, bias=False)),
                ("dense", SafeDenseBlock(128, [1, ksize], [128, 32], 4)),
                ("convf", nn.Conv2d(256, 256, 1, bias=False))
            ]))
            u1 = nn.Sequential(OrderedDict([("conva", nn.Conv2d(256, 64, ksize, padding=pad, bias=False))]))
            u0 = nn.Sequential(
                OrderedDict([("bn", nn.BatchNorm2d(64)), ("relu", nn.ReLU(True)), ("conv", nn.Conv2d(64, out_ch, 1))]))
            return nn.ModuleList([u3, u2, u1, u0])

        ks = 5 if mode == 'original' else 3
        cfg = ["np", "hv"] + (["tp_TTF-1", "tp_WT-1"] if nr_types else [])
        self.decoder = nn.ModuleDict({n: create_branch(nr_types if "tp" in n else 2, ks) for n in cfg})
        self.mtca = Directed_MTCA(64) if nr_types else None
        self.upsample2x = UpSample2x()
        self.weights_init()

    def forward(self, imgs):
        imgs = imgs / 255.0
        with torch.set_grad_enabled(not self.freeze):
            feats = self.backbone(imgs)
        d = self.adapter(feats)
        d[3] = self.conv_bot(d[3])

        u1_feats = {}
        for name, mods in self.decoder.items():
            sz3 = d[2].shape[2:]
            x = mods[0](F.interpolate(self.upsample2x(d[3]), size=sz3, mode='bilinear') + d[2])

            sz2 = d[1].shape[2:]
            x = mods[1](F.interpolate(self.upsample2x(x), size=sz2, mode='bilinear') + d[1])

            sz1 = d[0].shape[2:]
            u1_feats[name] = mods[2](F.interpolate(self.upsample2x(x), size=sz1, mode='bilinear') + d[0])

        if self.mtca:
            _, _, u1_feats["tp_TTF-1"], u1_feats["tp_WT-1"] = self.mtca(u1_feats["np"], u1_feats["hv"],
                                                                        u1_feats["tp_TTF-1"], u1_feats["tp_WT-1"])

        out = OrderedDict()
        for n, mods in self.decoder.items():
            out[n] = F.interpolate(mods[3](u1_feats[n]), (164, 164), mode='bilinear')
        return out


def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        raise ValueError("Unknown Model Mode %s" % mode)
    return ConvNeXt_Base_TripMTCA_HoVerNet(mode=mode, **kwargs)