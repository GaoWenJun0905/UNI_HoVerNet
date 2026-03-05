import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from misc.utils import center_pad_to_shape, cropping_center
from .utils import crop_to_shape, dice_loss, mse_loss, msge_loss, xentropy_loss, cost_xentropy_loss

from collections import OrderedDict

####
def train_step(batch_data, run_info):
    run_info, state_info = run_info
    loss_func_dict = {
        "bce": xentropy_loss,
        "dice": dice_loss,
        "mse": mse_loss,
        "msge": msge_loss,
        "cost": cost_xentropy_loss,
    }
    result_dict = {"EMA": {}}
    track_value = lambda name, value: result_dict["EMA"].update({name: value})

    model = run_info["net"]["desc"]
    optimizer = run_info["net"]["optimizer"]

    # 1. 准备数据
    imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]

    # 获取每一张图的任务类型
    task_mode_list = batch_data.get("IHC_type", ["unknown"] * len(imgs))
    task_mode_arr = np.array(task_mode_list)

    imgs = imgs.to("cuda").type(torch.float32).permute(0, 3, 1, 2).contiguous()
    true_np = true_np.to("cuda").type(torch.int64)
    true_hv = true_hv.to("cuda").type(torch.float32)

    # [关键] 生成 One-Hot 仅用于计算 Loss
    true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)

    true_dict = {
        "np": true_np_onehot,  # 这里存的是 One-Hot
        "hv": true_hv,
    }

    if model.module.nr_types is not None:
        true_tp = batch_data["tp_map"]
        true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)
        true_tp_onehot = F.one_hot(true_tp, num_classes=model.module.nr_types).type(torch.float32)

    model.train()
    model.zero_grad()

    # 2. 前向传播
    raw_pred_dict = model(imgs)

    pred_dict = OrderedDict()
    pred_dict["np"] = F.softmax(raw_pred_dict["np"].permute(0, 2, 3, 1).contiguous(), dim=-1)
    pred_dict["hv"] = raw_pred_dict["hv"].permute(0, 2, 3, 1).contiguous()

    # 3. 计算 Loss
    loss = 0
    loss_opts = run_info["net"]["extra_info"]["loss"]

    # --- A. 共享分支 Loss (NP, HV) ---
    for branch_name in ["np", "hv"]:
        if branch_name in loss_opts:
            for loss_name, loss_weight in loss_opts[branch_name].items():
                loss_func = loss_func_dict[loss_name]
                loss_args = [true_dict[branch_name], pred_dict[branch_name]]
                if loss_name == "msge":
                    loss_args.append(true_np_onehot[..., 1])

                term_loss = loss_func(*loss_args)
                track_value("loss_%s_%s" % (branch_name, loss_name), term_loss.cpu().item())
                loss += loss_weight * term_loss

    # --- B. 互斥分类头 Loss (分流处理) ---
    if model.module.nr_types is not None:
        tp_loss_opts = loss_opts.get("tp", {})
        task_map = {
            'TTF-1': 'tp_TTF-1',
            'WT-1': 'tp_WT-1'
        }

        for task_name, head_name in task_map.items():
            indices = np.where(task_mode_arr == task_name)[0]

            if len(indices) > 0:
                idx_tensor = torch.from_numpy(indices).to("cuda")

                # 提取预测值子集
                pred_all = raw_pred_dict[head_name].permute(0, 2, 3, 1).contiguous()
                pred_sub = torch.index_select(pred_all, 0, idx_tensor)
                pred_sub = F.softmax(pred_sub, dim=-1)

                # 提取真值子集
                true_sub = torch.index_select(true_tp_onehot, 0, idx_tensor)

                # 计算 Loss
                for loss_name, loss_weight in tp_loss_opts.items():
                    loss_func = loss_func_dict[loss_name]
                    term_loss = loss_func(true_sub, pred_sub)

                    log_key = "loss_tp_%s_%s" % (task_name, loss_name)
                    track_value(log_key, term_loss.cpu().item())
                    loss += loss_weight * term_loss

    track_value("overall_loss", loss.cpu().item())
    loss.backward()
    optimizer.step()

    # --- 4. 准备返回数据 (Visualization用) ---
    sample_indices = torch.randint(0, true_np.shape[0], (2,))

    # 构建混合 pred_tp 用于显示
    if model.module.nr_types is not None:
        B, H, W, C = pred_dict["np"].shape[0], pred_dict["np"].shape[1], pred_dict["np"].shape[2], model.module.nr_types
        mixed_pred_tp = torch.zeros((B, H, W, C), device="cuda")

        for task_name, head_name in task_map.items():
            indices = np.where(task_mode_arr == task_name)[0]
            if len(indices) > 0:
                idx_tensor = torch.from_numpy(indices).to("cuda")
                pred_all = raw_pred_dict[head_name].permute(0, 2, 3, 1).contiguous()
                pred_sub = torch.index_select(pred_all, 0, idx_tensor)
                pred_sub = F.softmax(pred_sub, dim=-1)
                mixed_pred_tp[idx_tensor] = pred_sub.detach()

    imgs = (imgs[sample_indices]).byte().permute(0, 2, 3, 1).contiguous().cpu().numpy()

    # [关键修复]
    # 这里的 True NP 必须使用 true_np (索引图, shape=[2,H,W]),
    # 不能用 true_dict["np"] (One-Hot图, shape=[2,H,W,2])
    res_np = (
        true_np[sample_indices].detach().cpu().numpy(),
        pred_dict["np"][sample_indices, ..., 1].detach().cpu().numpy()
    )

    res_hv = (
        true_dict["hv"][sample_indices].detach().cpu().numpy(),
        pred_dict["hv"][sample_indices].detach().cpu().numpy()
    )

    result_dict["raw"] = {
        "img": imgs,
        "np": res_np,
        "hv": res_hv,
    }

    if model.module.nr_types is not None:
        res_tp = (
            true_tp_onehot[sample_indices].detach().cpu().numpy(),
            mixed_pred_tp[sample_indices].cpu().numpy()
        )
        result_dict["raw"]["tp"] = res_tp

    return result_dict


####
def valid_step(batch_data, run_info):
    run_info, state_info = run_info
    ####
    model = run_info["net"]["desc"]
    model.eval()  # infer mode

    ####
    imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]

    # 1. 获取当前 Batch 中每一张图的任务类型
    # task_mode_list: ['TTF-1', 'WT-1', 'TTF-1', ...]
    task_mode_list = batch_data.get("IHC_type", ["unknown"] * len(imgs))
    task_mode_arr = np.array(task_mode_list)

    # 映射任务 ID 用于后续统计 (1=TTF-1, 2=WT-1)
    # 这一步非常重要，必须生成一个数组，不能是一个标量
    task_ids = np.zeros(len(imgs), dtype=np.int32)
    task_ids[task_mode_arr == 'TTF-1'] = 1
    task_ids[task_mode_arr == 'WT-1'] = 2

    imgs_gpu = imgs.to("cuda").type(torch.float32).permute(0, 3, 1, 2).contiguous()

    # HWC
    true_np = torch.squeeze(true_np).type(torch.int64)
    true_hv = torch.squeeze(true_hv).type(torch.float32)

    true_dict = {
        "np": true_np,
        "hv": true_hv,
    }

    if model.module.nr_types is not None:
        true_tp = batch_data["tp_map"]
        true_tp = torch.squeeze(true_tp).type(torch.int64)
        true_dict["tp"] = true_tp

    # --------------------------------------------------------------
    with torch.no_grad():
        # 2. 前向推理 (获取所有头的原始输出)
        raw_pred_dict = model(imgs_gpu)
        pred_dict = OrderedDict()

        # 处理共享分支
        pred_dict["np"] = raw_pred_dict["np"].permute(0, 2, 3, 1).contiguous()
        pred_dict["hv"] = raw_pred_dict["hv"].permute(0, 2, 3, 1).contiguous()

        # NP Softmax
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1]

        # 3. 处理互斥分类头 (分流合并策略)
        if model.module.nr_types is not None:
            # 创建一个全零的 Logits 容器 (B, H, W, nr_types)
            B, H, W = pred_dict["np"].shape
            C = model.module.nr_types
            mixed_tp_logits = torch.zeros((B, H, W, C), device="cuda", dtype=torch.float32)

            # 定义任务映射
            task_map = {
                'TTF-1': 'tp_TTF-1',
                'WT-1': 'tp_WT-1'
            }

            # 分别填空
            for task_name, head_name in task_map.items():
                # 找到属于该任务的图片索引
                indices = np.where(task_mode_arr == task_name)[0]

                if len(indices) > 0:
                    idx_tensor = torch.from_numpy(indices).to("cuda")

                    # 取出对应头的输出 (B, C, H, W)
                    head_out = raw_pred_dict[head_name]

                    # [关键点] 先用 index_select 提取子集！防止 shape mismatch
                    subset_out = torch.index_select(head_out, 0, idx_tensor)

                    # 变换维度 -> (Subset, H, W, nr_types)
                    subset_out = subset_out.permute(0, 2, 3, 1).contiguous()

                    # 填入混合容器
                    # mixed_tp_logits[idx_tensor] 的形状是 (Subset, H, W, C)
                    # subset_out 的形状也是 (Subset, H, W, C) -> 完美匹配
                    mixed_tp_logits[idx_tensor] = subset_out

            # 统一做 Softmax 和 Argmax
            # 验证阶段我们通常直接看 Argmax 后的分类图
            type_map = F.softmax(mixed_tp_logits, dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=False)
            type_map = type_map.type(torch.float32)

            pred_dict["tp"] = type_map

    # 4. 准备返回结果
    result_dict = {
        "raw": {
            "imgs": imgs.numpy(),
            "true_np": true_dict["np"].numpy(),
            "true_hv": true_dict["hv"].numpy(),
            "prob_np": pred_dict["np"].cpu().numpy(),
            "pred_hv": pred_dict["hv"].cpu().numpy(),
            "IHC_type": task_ids  # 返回数组 (B,)
        }
    }
    if model.module.nr_types is not None:
        result_dict["raw"]["true_tp"] = true_dict["tp"].numpy()
        result_dict["raw"]["pred_tp"] = pred_dict["tp"].cpu().numpy()

    return result_dict


####
# def infer_step(batch_data, model):
#
#     ####
#     patch_imgs = batch_data
#
#     patch_imgs_gpu = patch_imgs.to("cuda").type(torch.float32)  # to NCHW
#     patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()
#
#     ####
#     model.eval()  # infer mode
#
#     # --------------------------------------------------------------
#     with torch.no_grad():  # dont compute gradient
#         pred_dict = model(patch_imgs_gpu)
#         pred_dict = OrderedDict(
#             [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
#         )
#         pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
#         if "tp" in pred_dict:
#             type_map = F.softmax(pred_dict["tp"], dim=-1)
#             type_map = torch.argmax(type_map, dim=-1, keepdim=True)
#             type_map = type_map.type(torch.float32)
#             pred_dict["tp"] = type_map
#         pred_output = torch.cat(list(pred_dict.values()), -1)
#
#     # * Its up to user to define the protocol to process the raw output per step!
#     return pred_output.cpu().numpy()
# 20260122_GWJ_overlap没轮廓，但是debug都有东西
# import torch
# import torch.nn.functional as F
# from collections import OrderedDict
# import numpy as np
#
#
# def infer_step(batch_data, model):
#     # ----------------------------------------------------
#     # 1. 准备数据 (保持 0-255 原始输入)
#     # ----------------------------------------------------
#     patch_imgs = batch_data
#
#     # 不归一化，直接转 GPU Float32
#     patch_imgs_gpu = patch_imgs.to("cuda").type(torch.float32).permute(0, 3, 1, 2).contiguous()
#
#     # ----------------------------------------------------
#     # 2. 模型推理
#     # ----------------------------------------------------
#     model.eval()
#     with torch.no_grad():
#         raw_pred_dict = model(patch_imgs_gpu)
#
#         # 统一转为 (B, H, W, C)
#         pred_dict = OrderedDict(
#             [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in raw_pred_dict.items()]
#         )
#
#         final_dict = OrderedDict()
#
#         # --- A. NP (核概率) ---
#         # 原始逻辑: Softmax -> 取 Index 1
#         final_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
#
#         # --- B. HV (核梯度) ---
#         final_dict["hv"] = pred_dict["hv"]
#
#         # --- C. TP (双头严格逻辑融合) ---
#         # 目标: 0=背景, 1=腺癌(TTF-1+), 2=间皮(WT-1+), 3=其他(双阴性)
#         if "tp_TTF-1" in pred_dict and "tp_WT-1" in pred_dict:
#             # 1. 先对每个头单独做 Argmax，拿到确切的分类结果 (0, 1, 2)
#             # label_t: TTF-1 头的判断
#             # label_w: WT-1 头的判断
#             prob_t = F.softmax(pred_dict["tp_TTF-1"], dim=-1)
#             prob_w = F.softmax(pred_dict["tp_WT-1"], dim=-1)
#
#             label_t = torch.argmax(prob_t, dim=-1)  # (B, H, W)
#             label_w = torch.argmax(prob_w, dim=-1)  # (B, H, W)
#
#             # 2. 初始化最终图 (默认为 0 背景)
#             final_tp = torch.zeros_like(label_t, dtype=torch.float32)
#
#             # --- 规则执行 (注意顺序，后执行的会覆盖前面的) ---
#
#             # 规则 1: 两个头都认为是阴性(1) -> 类别 3 (其他)
#             # 逻辑: T=1 AND W=1
#             mask_other = (label_t == 1) & (label_w == 1)
#             final_tp[mask_other] = 3
#
#             # 规则 2: TTF-1 认为是阳性(2) -> 类别 1 (腺癌)
#             # 逻辑: T=2
#             mask_adeno = (label_t == 2)
#             final_tp[mask_adeno] = 1
#
#             # 规则 3: WT-1 认为是阳性(2) -> 类别 2 (间皮)
#             # 逻辑: W=2
#             mask_meso = (label_w == 2)
#             final_tp[mask_meso] = 2
#
#             # --- [冲突解决] 如果 T=2 且 W=2 怎么办？ ---
#             # 这种情况虽然少见，但必须处理。我们看谁的置信度(概率)更高。
#             mask_conflict = mask_adeno & mask_meso
#             if mask_conflict.any():
#                 # 取出冲突位置的概率值
#                 p_t_pos = prob_t[..., 2]
#                 p_w_pos = prob_w[..., 2]
#
#                 # 如果 TTF-1 的概率比 WT-1 高，改回 1 (因为上面规则3先把这里设为2了)
#                 # 只有在冲突区域，且 T > W 时，才改回 1
#                 conflict_t_wins = mask_conflict & (p_t_pos > p_w_pos)
#                 final_tp[conflict_t_wins] = 1
#
#             # 最终结果维度调整为 (B, H, W, 1)
#             final_dict["tp"] = final_tp.unsqueeze(-1)
#
#         # D. 拼接输出
#         pred_output = torch.cat(list(final_dict.values()), -1)
#
#     return pred_output.cpu().numpy()

# 20260123_GWJ_预测的概率不太行
# import torch
# import torch.nn.functional as F
# from collections import OrderedDict
# import numpy as np
#
#
# def infer_step(batch_data, model):
#     # 1. 准备数据
#     patch_imgs = batch_data
#     patch_imgs_gpu = patch_imgs.to("cuda").type(torch.float32).permute(0, 3, 1, 2).contiguous()
#
#     # 2. 推理
#     model.eval()
#     with torch.no_grad():
#         raw_pred_dict = model(patch_imgs_gpu)
#         pred_dict = OrderedDict(
#             [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in raw_pred_dict.items()]
#         )
#
#         # --- A. NP [Ch0, Ch1] ---
#         np_prob = F.softmax(pred_dict["np"], dim=-1)  # (B, H, W, 2)
#
#         # --- B. HV [Ch2, Ch3] ---
#         hv_map = pred_dict["hv"]  # (B, H, W, 2)
#
#         # --- C. TP [Ch4, Ch5, Ch6, Ch7] (核心修改) ---
#         if "tp_TTF-1" in pred_dict and "tp_WT-1" in pred_dict:
#             # 1. 获取概率
#             prob_t = F.softmax(pred_dict["tp_TTF-1"], dim=-1)
#             prob_w = F.softmax(pred_dict["tp_WT-1"], dim=-1)
#
#             # 2. 硬分类 (Strict Logic)
#             label_t = torch.argmax(prob_t, dim=-1)
#             label_w = torch.argmax(prob_w, dim=-1)
#
#             # 初始化整数图
#             tp_int = torch.zeros_like(label_t, dtype=torch.long)
#
#             # 规则执行
#             tp_int[(label_t == 1) & (label_w == 1)] = 3  # 双阴 -> 3
#             tp_int[label_t == 2] = 1  # TTF-1 -> 1
#             tp_int[label_w == 2] = 2  # WT-1 -> 2
#
#             # 冲突解决
#             mask_conflict = (label_t == 2) & (label_w == 2)
#             if mask_conflict.any():
#                 conflict_t_wins = mask_conflict & (prob_t[..., 2] > prob_w[..., 2])
#                 tp_int[conflict_t_wins] = 1
#
#             # [关键] 转为 One-Hot (4通道)
#             # shape: (B, H, W) -> (B, H, W, 4)
#             tp_map = F.one_hot(tp_int, num_classes=4).float()
#
#         else:
#             # 没有分类头，全填为背景(类0)
#             B, H, W, _ = np_prob.shape
#             tp_map = torch.zeros((B, H, W, 4), device="cuda", dtype=torch.float32)
#             tp_map[..., 0] = 1.0  # Channel 0 = 1.0
#
#         # 拼接: [NP(2) + HV(2) + TP(4)] = 8 Channels
#         # 顺序:
#         # 0:NP_BG, 1:NP_FG
#         # 2:HVx, 3:HVy
#         # 4:Cls0, 5:Cls1, 6:Cls2, 7:Cls3
#         pred_output = torch.cat([np_prob, hv_map, tp_map], dim=-1)
#
#     return pred_output.cpu().numpy()
# 20260125_GWJ_Test，不好权衡两个分类头的结果，效果不好
# import torch
# import torch.nn.functional as F
# from collections import OrderedDict
# import numpy as np
#
#
# def infer_step(batch_data, model):
#     # 1. 准备数据
#     patch_imgs = batch_data
#     patch_imgs_gpu = patch_imgs.to("cuda").type(torch.float32).permute(0, 3, 1, 2).contiguous()
#
#     # 2. 推理
#     model.eval()
#     with torch.no_grad():
#         raw_pred_dict = model(patch_imgs_gpu)
#         pred_dict = OrderedDict(
#             [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in raw_pred_dict.items()]
#         )
#
#         # --- A. NP [Ch0, Ch1] ---
#         np_prob = F.softmax(pred_dict["np"], dim=-1)  # (B, H, W, 2)
#
#         # --- B. HV [Ch2, Ch3] ---
#         hv_map = pred_dict["hv"]  # (B, H, W, 2)
#
#         # --- C. TP [Ch4, Ch5, Ch6, Ch7] (软概率融合版) ---
#         if "tp_TTF-1" in pred_dict and "tp_WT-1" in pred_dict:
#             # 1. 获取概率
#             prob_t = F.softmax(pred_dict["tp_TTF-1"], dim=-1)
#             prob_w = F.softmax(pred_dict["tp_WT-1"], dim=-1)
#
#             # --- 改进点：引入互斥惩罚 (Exclusion Penalty) ---
#
#             # [Score 0: 背景]
#             # 保持 0，或者给极小值，依赖 NP 分支做检测
#             score_bg = torch.zeros_like(prob_t[..., 0])
#
#             # [Score 1: 腺癌]
#             # 逻辑：TTF-1 必须是阳性(Ch2)，同时 WT-1 必须是阴性(Ch1)
#             # 使用几何平均 (sqrt(a*b)) 来平衡两个概率
#             # 效果：如果 WT-1 误报阳性，它的阴性概率(Ch1)就会变低，从而拉低腺癌的总分
#             p_t_pos = prob_t[..., 2]
#             p_w_neg = prob_w[..., 1]
#             score_adeno = torch.sqrt(p_t_pos * p_w_neg + 1e-6)
#
#             # [Score 2: 间皮]
#             # 逻辑：WT-1 必须是阳性(Ch2)，同时 TTF-1 必须是阴性(Ch1)
#             p_w_pos = prob_w[..., 2]
#             p_t_neg = prob_t[..., 1]
#             score_meso = torch.sqrt(p_w_pos * p_t_neg + 1e-6)
#
#             # [Score 3: 其他/双阴性]
#             # 逻辑：TTF-1 是阴性(Ch1) 且 WT-1 是阴性(Ch1)
#             # 只有两个头都确认是阴性，这个分才高
#             score_other = torch.sqrt(p_t_neg * p_w_neg + 1e-6)
#
#             # 3. 堆叠 & 4. 决胜
#             merged_scores = torch.stack([score_bg, score_adeno, score_meso, score_other], dim=-1)
#             final_cls = torch.argmax(merged_scores, dim=-1)
#
#             # 5. 转为 One-Hot (为了兼容后处理格式)
#             tp_map = F.one_hot(final_cls, num_classes=4).float()
#
#         else:
#             # 没有分类头，全填为背景
#             B, H, W, _ = np_prob.shape
#             tp_map = torch.zeros((B, H, W, 4), device="cuda", dtype=torch.float32)
#             tp_map[..., 0] = 1.0
#
#         # 拼接: [NP(2) + HV(2) + TP(4)] = 8 Channels
#         # 顺序:
#         # 0:NP_BG, 1:NP_FG
#         # 2:HVx, 3:HVy
#         # 4:Cls0, 5:Cls1, 6:Cls2, 7:Cls3
#         pred_output = torch.cat([np_prob, hv_map, tp_map], dim=-1)
#
#     return pred_output.cpu().numpy()
# 20260126_GWJ_test
def infer_step(batch_data, model):
    # 1. 准备数据
    patch_imgs = batch_data
    patch_imgs_gpu = patch_imgs.to("cuda").type(torch.float32).permute(0, 3, 1, 2).contiguous()

    # 2. 推理
    model.eval()
    with torch.no_grad():
        raw_pred_dict = model(patch_imgs_gpu)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in raw_pred_dict.items()]
        )

        # --- A. NP [Ch0, Ch1] ---
        np_prob = F.softmax(pred_dict["np"], dim=-1)

        # --- B. HV [Ch2, Ch3] ---
        hv_map = pred_dict["hv"]

        # --- C. 分离的分类头输出 ---
        # 我们不在这里做决定，而是把原始概率全部传出去
        if "tp_TTF-1" in pred_dict and "tp_WT-1" in pred_dict:
            # TTF-1 概率 (B, H, W, 3) -> [Bg, Neg, Pos]
            prob_t = F.softmax(pred_dict["tp_TTF-1"], dim=-1)

            # WT-1 概率 (B, H, W, 3) -> [Bg, Neg, Pos]
            prob_w = F.softmax(pred_dict["tp_WT-1"], dim=-1)

            # 拼接: NP(2) + HV(2) + TTF(3) + WT(3) = 10 通道
            pred_output = torch.cat([np_prob, hv_map, prob_t, prob_w], dim=-1)
        else:
            # 兼容没有分类头的情况 (补零)
            B, H, W, _ = np_prob.shape
            dummy = torch.zeros((B, H, W, 6), device="cuda", dtype=torch.float32)
            pred_output = torch.cat([np_prob, hv_map, dummy], dim=-1)

    return pred_output.cpu().numpy()


####
def viz_step_output(raw_data, nr_types=None):
    """
    `raw_data` will be implicitly provided in the similar format as the 
    return dict from train/valid step, but may have been accumulated across N running step
    """

    imgs = raw_data["img"]
    true_np, pred_np = raw_data["np"]
    true_hv, pred_hv = raw_data["hv"]
    if nr_types is not None:
        true_tp, pred_tp = raw_data["tp"]

    aligned_shape = [list(imgs.shape), list(true_np.shape), list(pred_np.shape)]
    # 20260116_GWJ
    # aligned_shape = np.min(np.array(aligned_shape), axis=0)[1:3]
    shape_list = [s[1:3] for s in aligned_shape]
    aligned_shape = np.min(np.array(shape_list), axis=0)

    cmap = plt.get_cmap("jet")

    def colorize(ch, vmin, vmax):
        """
        Will clamp value value outside the provided range to vmax and vmin
        """
        ch = np.squeeze(ch.astype("float32"))
        ch[ch > vmax] = vmax  # clamp value
        ch[ch < vmin] = vmin
        ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
        # take RGB from RGBA heat map
        ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
        # ch_cmap = center_pad_to_shape(ch_cmap, aligned_shape)
        return ch_cmap

    viz_list = []
    for idx in range(imgs.shape[0]):
        # img = center_pad_to_shape(imgs[idx], aligned_shape)
        img = cropping_center(imgs[idx], aligned_shape)

        true_viz_list = [img]
        # cmap may randomly fails if of other types
        true_viz_list.append(colorize(true_np[idx], 0, 1))
        true_viz_list.append(colorize(true_hv[idx][..., 0], -1, 1))
        true_viz_list.append(colorize(true_hv[idx][..., 1], -1, 1))
        if nr_types is not None:  # TODO: a way to pass through external info
            true_viz_list.append(colorize(true_tp[idx], 0, nr_types))
        true_viz_list = np.concatenate(true_viz_list, axis=1)

        pred_viz_list = [img]
        # cmap may randomly fails if of other types
        pred_viz_list.append(colorize(pred_np[idx], 0, 1))
        pred_viz_list.append(colorize(pred_hv[idx][..., 0], -1, 1))
        pred_viz_list.append(colorize(pred_hv[idx][..., 1], -1, 1))
        if nr_types is not None:
            pred_viz_list.append(colorize(pred_tp[idx], 0, nr_types))
        pred_viz_list = np.concatenate(pred_viz_list, axis=1)

        viz_list.append(np.concatenate([true_viz_list, pred_viz_list], axis=0))
    viz_list = np.concatenate(viz_list, axis=0)
    return viz_list


####
from itertools import chain


# def proc_valid_step_output(raw_data, nr_types=None):
#     # TODO: add auto populate from main state track list
#     track_dict = {"scalar": {}, "image": {}}
#
#     def track_value(name, value, vtype):
#         return track_dict[vtype].update({name: value})
#
#     def _dice_info(true, pred, label):
#         true = np.array(true == label, np.int32)
#         pred = np.array(pred == label, np.int32)
#         inter = (pred * true).sum()
#         total = (pred + true).sum()
#         return inter, total
#
#     over_inter = 0
#     over_total = 0
#     over_correct = 0
#     prob_np = raw_data["prob_np"]
#     true_np = raw_data["true_np"]
#     for idx in range(len(raw_data["true_np"])):
#         patch_prob_np = prob_np[idx]
#         patch_true_np = true_np[idx]
#         patch_pred_np = np.array(patch_prob_np > 0.5, dtype=np.int32)
#         inter, total = _dice_info(patch_true_np, patch_pred_np, 1)
#         correct = (patch_pred_np == patch_true_np).sum()
#         over_inter += inter
#         over_total += total
#         over_correct += correct
#     nr_pixels = len(true_np) * np.size(true_np[0])
#     acc_np = over_correct / nr_pixels
#     dice_np = 2 * over_inter / (over_total + 1.0e-8)
#     track_value("np_acc", acc_np, "scalar")
#     track_value("np_dice", dice_np, "scalar")
#
#     # * TP statistic
#     if nr_types is not None:
#         pred_tp = raw_data["pred_tp"]
#         true_tp = raw_data["true_tp"]
#         for type_id in range(0, nr_types):
#             over_inter = 0
#             over_total = 0
#             for idx in range(len(raw_data["true_np"])):
#                 patch_pred_tp = pred_tp[idx]
#                 patch_true_tp = true_tp[idx]
#                 inter, total = _dice_info(patch_true_tp, patch_pred_tp, type_id)
#                 over_inter += inter
#                 over_total += total
#             dice_tp = 2 * over_inter / (over_total + 1.0e-8)
#             track_value("tp_dice_%d" % type_id, dice_tp, "scalar")
#
#     # * HV regression statistic
#     pred_hv = raw_data["pred_hv"]
#     true_hv = raw_data["true_hv"]
#
#     over_squared_error = 0
#     for idx in range(len(raw_data["true_np"])):
#         patch_pred_hv = pred_hv[idx]
#         patch_true_hv = true_hv[idx]
#         squared_error = patch_pred_hv - patch_true_hv
#         squared_error = squared_error * squared_error
#         over_squared_error += squared_error.sum()
#     mse = over_squared_error / nr_pixels
#     track_value("hv_mse", mse, "scalar")
#
#     # *
#     imgs = raw_data["imgs"]
#     selected_idx = np.random.randint(0, len(imgs), size=(8,)).tolist()
#     imgs = np.array([imgs[idx] for idx in selected_idx])
#     true_np = np.array([true_np[idx] for idx in selected_idx])
#     true_hv = np.array([true_hv[idx] for idx in selected_idx])
#     prob_np = np.array([prob_np[idx] for idx in selected_idx])
#     pred_hv = np.array([pred_hv[idx] for idx in selected_idx])
#     viz_raw_data = {"img": imgs, "np": (true_np, prob_np), "hv": (true_hv, pred_hv)}
#
#     if nr_types is not None:
#         true_tp = np.array([true_tp[idx] for idx in selected_idx])
#         pred_tp = np.array([pred_tp[idx] for idx in selected_idx])
#         viz_raw_data["tp"] = (true_tp, pred_tp)
#     viz_fig = viz_step_output(viz_raw_data, nr_types)
#     track_dict["image"]["output"] = viz_fig
#
#     return track_dict
def proc_valid_step_output(raw_data, nr_types=None):
    track_dict = {"scalar": {}, "image": {}}

    def track_value(name, value, vtype):
        return track_dict[vtype].update({name: value})

    # 辅助函数：计算单个 Batch 的交集和总数
    def _batch_dice_stat(true, pred, label):
        true_mask = (true == label)
        pred_mask = (pred == label)
        inter = (true_mask & pred_mask).sum()
        total = true_mask.sum() + pred_mask.sum()
        return inter, total

    # =============================================================
    # 1. 初始化累加器 (Accumulators)
    # =============================================================

    # NP (核像素) 累加器
    np_stats = {"inter": 0, "total": 0, "correct_pixels": 0, "total_pixels": 0}

    # HV (回归) 累加器
    hv_stats = {"mse_sum": 0, "pixel_count": 0}

    # TP (分类) 累加器: 结构为 { "TTF-1": {0: [inter, total], 1: ...}, "WT-1": ... }
    tp_stats = {}
    if nr_types is not None:
        tasks = ["TTF-1", "WT-1"]
        for t in tasks:
            tp_stats[t] = {}
            for cls in range(nr_types):
                # [intersection_sum, total_sum, true_pixel_count, pred_pixel_count]
                tp_stats[t][cls] = [0, 0, 0, 0]

                # =============================================================
    # 2. 逐 Batch 遍历 (流式处理，内存占用极低)
    # =============================================================

    # 获取 Batch 数量
    num_batches = len(raw_data["prob_np"])

    # 既然是一个列表，我们直接用 zip 遍历所有数据
    # 注意：raw_data 里的每个元素都是一个 list，长度为 num_batches

    # 提取 task_ids 列表 (兼容性处理)
    raw_ids_list = raw_data.get("task_ids", raw_data.get("IHC_type", [None] * num_batches))

    for i in range(num_batches):
        # 1. 获取当前 Batch 的数据
        batch_prob_np = raw_data["prob_np"][i]  # (B, H, W)
        batch_true_np = raw_data["true_np"][i]  # (B, H, W)
        batch_pred_hv = raw_data["pred_hv"][i]  # (B, H, W, 2)
        batch_true_hv = raw_data["true_hv"][i]  # (B, H, W, 2)

        # 获取当前 Batch 的 Task ID
        # 注意：valid_step 返回的 task_ids 是 numpy array (B,)
        # 我们取第一个元素即可，因为同一个 Batch 任务通常是一样的
        if raw_ids_list[i] is not None:
            batch_ids = np.atleast_1d(raw_ids_list[i])
            # 假设一个 batch 里任务是一样的，取第一个
            # 如果不想假设，也可以在这个 batch 内部再循环，但效率低
            current_task_id = batch_ids[0]
        else:
            current_task_id = 0  # Unknown

        # 映射 ID 到名称
        if current_task_id == 1:
            task_name = "TTF-1"
        elif current_task_id == 2:
            task_name = "WT-1"
        else:
            task_name = None

        # --- 计算 NP 指标 (累加) ---
        batch_pred_np = (batch_prob_np > 0.5).astype(np.int32)
        batch_true_np = batch_true_np.astype(np.int32)

        inter, total = _batch_dice_stat(batch_true_np, batch_pred_np, 1)
        np_stats["inter"] += inter
        np_stats["total"] += total
        np_stats["correct_pixels"] += (batch_pred_np == batch_true_np).sum()
        np_stats["total_pixels"] += batch_true_np.size

        # --- 计算 HV 指标 (累加) ---
        sq_err = (batch_pred_hv - batch_true_hv) ** 2
        hv_stats["mse_sum"] += sq_err.sum()
        hv_stats["pixel_count"] += batch_true_np.size  # HV 也是逐像素

        # --- 计算 TP 指标 (累加) ---
        if nr_types is not None and task_name is not None:
            batch_pred_tp = raw_data["pred_tp"][i]
            batch_true_tp = raw_data["true_tp"][i]

            # 针对该任务的每一个类别进行统计
            for cls_id in range(nr_types):
                inter, total = _batch_dice_stat(batch_true_tp, batch_pred_tp, cls_id)

                # 统计像素用于诊断
                true_pix_count = (batch_true_tp == cls_id).sum()
                pred_pix_count = (batch_pred_tp == cls_id).sum()

                # 更新累加器
                tp_stats[task_name][cls_id][0] += inter
                tp_stats[task_name][cls_id][1] += total
                tp_stats[task_name][cls_id][2] += true_pix_count
                tp_stats[task_name][cls_id][3] += pred_pix_count

    # =============================================================
    # 3. 最终结算 (Final Calculation)
    # =============================================================

    # NP 结算
    np_dice = 2 * np_stats["inter"] / (np_stats["total"] + 1e-8)
    np_acc = np_stats["correct_pixels"] / (np_stats["total_pixels"] + 1e-8)
    track_value("np_dice", np_dice, "scalar")
    track_value("np_acc", np_acc, "scalar")

    # HV 结算
    hv_mse = hv_stats["mse_sum"] / (hv_stats["pixel_count"] + 1e-8)
    track_value("hv_mse", hv_mse, "scalar")

    # TP 结算
    if nr_types is not None:
        for t_name in ["TTF-1", "WT-1"]:
            for cls_id in range(nr_types):
                stats = tp_stats[t_name][cls_id]
                inter_sum = stats[0]
                total_sum = stats[1]
                true_pix_sum = stats[2]
                pred_pix_sum = stats[3]

                final_dice = 2 * inter_sum / (total_sum + 1e-8)

                # [诊断打印]
                if cls_id == 2 and final_dice == 0:
                    print(f"--- [Diag {t_name} Class 2] Total True: {true_pix_sum}, Total Pred: {pred_pix_sum} ---")

                track_value(f"tp_dice_{cls_id}_{t_name}", final_dice, "scalar")

    # =============================================================
    # 4. 可视化 (安全采样)
    # =============================================================
    # 我们不需要拼接所有图片，只需要从第 1 个 Batch 里取前 8 张图即可
    # 这样既快又安全
    try:
        # 取第一个 batch 的数据 (通常 valid batch size >= 4, 可能需要取两个 batch)
        viz_imgs = []
        viz_true_np = []
        viz_prob_np = []
        viz_true_hv = []
        viz_pred_hv = []
        viz_true_tp = []
        viz_pred_tp = []

        # 简单的循环收集前 8 张图
        collected = 0
        for i in range(num_batches):
            bs = len(raw_data["imgs"][i])
            if collected >= 8: break

            # 截取需要的数量
            need = min(8 - collected, bs)

            viz_imgs.append(raw_data["imgs"][i][:need])
            viz_true_np.append(raw_data["true_np"][i][:need])
            viz_prob_np.append(raw_data["prob_np"][i][:need])
            viz_true_hv.append(raw_data["true_hv"][i][:need])
            viz_pred_hv.append(raw_data["pred_hv"][i][:need])
            if nr_types is not None:
                viz_true_tp.append(raw_data["true_tp"][i][:need])
                viz_pred_tp.append(raw_data["pred_tp"][i][:need])

            collected += need

        # 只有这里需要拼接一小部分数据用于画图
        if collected > 0:
            viz_raw_data = {
                "img": np.concatenate(viz_imgs, axis=0),
                "np": (np.concatenate(viz_true_np, axis=0), np.concatenate(viz_prob_np, axis=0)),
                "hv": (np.concatenate(viz_true_hv, axis=0), np.concatenate(viz_pred_hv, axis=0))
            }
            if nr_types is not None:
                viz_raw_data["tp"] = (np.concatenate(viz_true_tp, axis=0), np.concatenate(viz_pred_tp, axis=0))

            viz_fig = viz_step_output(viz_raw_data, nr_types)
            track_dict["image"]["output"] = viz_fig

    except Exception as e:
        print(f"[Viz Error] Skip visualization: {e}")

    return track_dict
