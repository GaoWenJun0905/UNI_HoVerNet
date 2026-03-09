import math
import numpy as np

import torch
import torch.nn.functional as F

from matplotlib import cm


####
def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image.

    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`
        
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x


####
def crop_to_shape(x, y, data_format="NCHW"):
    """Centre crop x so that x has shape of y. y dims must be smaller than x dims.

    Args:
        x: input array
        y: array with desired shape.

    """
    assert (
        y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1]
    ), "Ensure that y dimensions are smaller than x dimensions!"

    x_shape = x.size()
    y_shape = y.size()
    if data_format == "NCHW":
        crop_shape = (x_shape[2] - y_shape[2], x_shape[3] - y_shape[3])
    else:
        crop_shape = (x_shape[1] - y_shape[1], x_shape[2] - y_shape[2])
    return crop_op(x, crop_shape, data_format)


####
def xentropy_loss(true, pred, reduction="mean"):
    """Cross entropy loss. Assumes NHWC!

    Args:
        pred: prediction array
        true: ground truth array
    
    Returns:
        cross entropy loss

    """
    epsilon = 10e-8
    # scale preds so that the class probs of each sample sum to 1
    pred = pred / torch.sum(pred, -1, keepdim=True)
    # manual computation of crossentropy
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
    loss = loss.mean() if reduction == "mean" else loss.sum()
    return loss


# 20260214_GWJ
def focal_loss(true, pred, gamma=2.0, alpha=0.25, reduction="mean"):
    """
    专门为 HoVer-Net 的 NHWC 输出设计的 Focal Loss。
    Args:
        true: One-hot 标签 [N, H, W, nr_types]
        pred: Softmax 后的概率 [N, H, W, nr_types] (注意：你的代码里 pred 已经做了 softmax)
    """
    epsilon = 1e-8
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)

    # 计算 Focal Weight: (1 - p_t)^gamma
    # 由于 true 是 one-hot，乘积后只剩下目标类别的 (1-p)
    focal_weight = torch.pow(1.0 - pred, gamma)

    # 结合 Cross Entropy
    loss = -alpha * focal_weight * true * torch.log(pred)

    loss = torch.sum(loss, dim=-1)  # 合并类别通道
    return loss.mean() if reduction == "mean" else loss.sum()


# 20260215_GWJ_Updated: 解决误判过多的分类损失函数
def bah_loss(true, pred, gamma=2.0, reduction="mean"):
    """
    BAH-Loss: 针对病理图像优化的平衡自适应损失。
    通过 alpha_weights 压低背景(0.1)，提高对 TTF-1/WT-1 (0.45) 的关注度。
    """
    # [背景, 类别1, 类别2, ...]
    # 假设你的 nr_types 包含背景在内共 3 类
    device = pred.device
    alpha_weights = torch.tensor([0.1, 0.45, 0.45], device=device)

    epsilon = 1e-8
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)

    # 1. 计算 CE 部分
    ce_loss = -true * torch.log(pred)

    # 2. 计算 Focal 权重因子 (1-p)^gamma
    focal_weight = torch.pow(1.0 - pred, gamma)

    # 3. 结合权重
    # alpha_weights 对各通道进行加权
    loss = alpha_weights * focal_weight * ce_loss

    loss = torch.sum(loss, dim=-1)  # 合并类别通道
    return loss.mean() if reduction == "mean" else loss.sum()

# 20260227_GWJ
def cost_sensitive_loss(input, target, M):
    target = torch.argmax(target, dim=-1, keepdim=False)
    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))
    device = input.device
    M = M.to(device)

    return torch.sum((M[target, :] * input.float()), dim=-1, keepdim=True)
def cost_xentropy_loss(true, pred, reduction="mean"):
    """Cross Sensitive loss. Assumes NHWC!

    Args:
        pred: prediction array
        true: ground truth array

    Returns:
        Cross Sensitive loss

    """
    N, H, W, C = pred.size()

    lambd = 0.2


    # M = np.array([
    #     [0, 1, 1, 1, 1],
    #     [2, 0, 2, 2, 2],
    #     [2, 2, 0, 2, 2],
    #     [10, 10, 10, 0, 10],
    #     [10, 10, 10, 10, 0]
    # ], dtype=np.float)

    # 20250227_GWJ:20260227-1训练用的矩阵
    # M = np.array([
    #     [0, 1, 1],
    #     [2, 0, 2],
    #     [10, 10, 0]
    # ])
    # 20250301_GWJ:20260301-1训练用的矩阵
    M = np.array([
        [0, 1, 1],
        [2, 0, 2],
        [20, 20, 0]
    ])

    # 20250115_GWJ:20260116-1训练用的矩阵
    # M = np.array([
    #     [0, 1, 1],
    #     [2, 0, 2],
    #     [20, 20, 0]
    # ])


    # M = M/M.sum()
    M = torch.from_numpy(M)

    # print("M",M)

    # if C == 5:
    if C == 3:
        costsensitive_loss = cost_sensitive_loss(pred, true, M)  # costsensitive_loss.shape torch.Size([8, 256, 256])
        costsensitive_loss = lambd * costsensitive_loss
        epsilon = 10e-8
        # scale preds so that the class probs of each sample sum to 1
        pred = pred / torch.sum(pred, -1, keepdim=True)
        # manual computation of crossentropy
        pred = torch.clamp(pred, epsilon, 1.0 - epsilon)

        loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
        loss = (costsensitive_loss + loss).mean() if reduction == "mean" else (costsensitive_loss + loss).sum()
    else:

        epsilon = 10e-8
        # scale preds so that the class probs of each sample sum to 1
        pred = pred / torch.sum(pred, -1, keepdim=True)
        pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
        loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
        loss = loss.mean() if reduction == "mean" else loss.sum()
        return loss

    return loss

####
def dice_loss(true, pred, smooth=1e-3):
    """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
    inse = torch.sum(pred * true, (0, 1, 2))
    l = torch.sum(pred, (0, 1, 2))
    r = torch.sum(true, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = torch.sum(loss)
    return loss


####
def mse_loss(true, pred):
    """Calculate mean squared error loss.

    Args:
        true: ground truth of combined horizontal
              and vertical maps
        pred: prediction of combined horizontal
              and vertical maps 
    
    Returns:
        loss: mean squared error

    """
    loss = pred - true
    loss = (loss * loss).mean()
    return loss


####
def msge_loss(true, pred, focus):
    """Calculate the mean squared error of the gradients of 
    horizontal and vertical map predictions. Assumes 
    channel 0 is Vertical and channel 1 is Horizontal.

    Args:
        true:  ground truth of combined horizontal
               and vertical maps
        pred:  prediction of combined horizontal
               and vertical maps 
        focus: area where to apply loss (we only calculate
                the loss within the nuclei)
    
    Returns:
        loss:  mean squared error of gradients

    """

    def get_sobel_kernel(size):
        """Get sobel kernel with a given size."""
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    ####
    def get_gradient_hv(hv):
        """For calculating gradient."""
        kernel_h, kernel_v = get_sobel_kernel(5)
        kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
        kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

        h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
        v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

        # can only apply in NCHW mode
        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
        return dhv

    focus = (focus[..., None]).float()  # assume input NHW
    focus = torch.cat([focus, focus], axis=-1)
    true_grad = get_gradient_hv(true)
    pred_grad = get_gradient_hv(pred)
    loss = pred_grad - true_grad
    loss = focus * (loss * loss)
    # artificial reduce_mean with focused region
    loss = loss.sum() / (focus.sum() + 1.0e-8)
    return loss
