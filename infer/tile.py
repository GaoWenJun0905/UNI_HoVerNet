import logging
import multiprocessing
from multiprocessing import Lock, Pool

multiprocessing.set_start_method("spawn", True)  # ! must be at top for VScode debugging
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import pathlib
import pickle
import re
import sys
import warnings
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, as_completed, wait
from functools import reduce
from importlib import import_module
from multiprocessing import Lock, Pool

import cv2
import numpy as np
import psutil
import scipy.io as sio
import torch
import torch.utils.data as data
import tqdm
from dataloader.infer_loader import SerializeArray, SerializeFileList
from misc.utils import (
    color_deconvolution,
    cropping_center,
    get_bounding_box,
    log_debug,
    log_info,
    rm_n_mkdir,
)
from misc.viz_utils import colorize, visualize_instances_dict
from skimage import color

import convert_format
from . import base


####
def _prepare_patching(img, window_size, mask_size, return_src_top_corner=False):
    """Prepare patch information for tile processing.
    
    Args:
        img: original input image
        window_size: input patch size
        mask_size: output patch size
        return_src_top_corner: whether to return coordiante information for top left corner of img
        
    """

    win_size = window_size
    msk_size = step_size = mask_size

    def get_last_steps(length, msk_size, step_size):
        nr_step = math.ceil((length - msk_size) / step_size)
        last_step = (nr_step + 1) * step_size
        return int(last_step), int(nr_step + 1)

    im_h = img.shape[0]
    im_w = img.shape[1]

    last_h, _ = get_last_steps(im_h, msk_size, step_size)
    last_w, _ = get_last_steps(im_w, msk_size, step_size)

    diff = win_size - step_size
    padt = padl = diff // 2
    padb = last_h + win_size - im_h
    padr = last_w + win_size - im_w

    img = np.pad(img, ((padt, padb), (padl, padr), (0, 0)), "reflect")

    # generating subpatches index from orginal
    coord_y = np.arange(0, last_h, step_size, dtype=np.int32)
    coord_x = np.arange(0, last_w, step_size, dtype=np.int32)
    row_idx = np.arange(0, coord_y.shape[0], dtype=np.int32)
    col_idx = np.arange(0, coord_x.shape[0], dtype=np.int32)
    coord_y, coord_x = np.meshgrid(coord_y, coord_x)
    row_idx, col_idx = np.meshgrid(row_idx, col_idx)
    coord_y = coord_y.flatten()
    coord_x = coord_x.flatten()
    row_idx = row_idx.flatten()
    col_idx = col_idx.flatten()
    #
    patch_info = np.stack([coord_y, coord_x, row_idx, col_idx], axis=-1)
    if not return_src_top_corner:
        return img, patch_info
    else:
        return img, patch_info, [padt, padl]


####
def _post_process_patches(
    post_proc_func, post_proc_kwargs, patch_info, image_info, overlay_kwargs,
):
    """Apply post processing to patches.
    
    Args:
        post_proc_func: post processing function to use
        post_proc_kwargs: keyword arguments used in post processing function
        patch_info: patch data and associated information
        image_info: input image data and associated information
        overlay_kwargs: overlay keyword arguments

    """
    # re-assemble the prediction, sort according to the patch location within the original image
    patch_info = sorted(patch_info, key=lambda x: [x[0][0], x[0][1]])
    patch_info, patch_data = zip(*patch_info)

    src_shape = image_info["src_shape"]
    src_image = image_info["src_image"]

    patch_shape = np.squeeze(patch_data[0]).shape
    ch = 1 if len(patch_shape) == 2 else patch_shape[-1]
    axes = [0, 2, 1, 3, 4] if ch != 1 else [0, 2, 1, 3]

    nr_row = max([x[2] for x in patch_info]) + 1
    nr_col = max([x[3] for x in patch_info]) + 1
    pred_map = np.concatenate(patch_data, axis=0)
    pred_map = np.reshape(pred_map, (nr_row, nr_col) + patch_shape)
    pred_map = np.transpose(pred_map, axes)
    pred_map = np.reshape(
        pred_map, (patch_shape[0] * nr_row, patch_shape[1] * nr_col, ch)
    )
    # crop back to original shape
    pred_map = np.squeeze(pred_map[: src_shape[0], : src_shape[1]])

    # * Implicit protocol
    # * a prediction map with instance of ID 1-N
    # * and a dict contain the instance info, access via its ID
    # * each instance may have type
    pred_inst, inst_info_dict = post_proc_func(pred_map, **post_proc_kwargs)

    overlaid_img = visualize_instances_dict(
        src_image.copy(), inst_info_dict, **overlay_kwargs
    )

    return image_info["name"], pred_map, pred_inst, inst_info_dict, overlaid_img


class InferManager(base.InferManager):
    """Run inference on tiles."""

    ####
    def process_file_list(self, run_args):
        """
        Process a single image tile < 5000x5000 in size.
        """
        for variable, value in run_args.items():
            self.__setattr__(variable, value)
        assert self.mem_usage < 1.0 and self.mem_usage > 0.0

        # * depend on the number of samples and their size, this may be less efficient
        patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
        file_path_list = glob.glob(patterning("%s/*" % self.input_dir))
        file_path_list.sort()  # ensure same order
        assert len(file_path_list) > 0, 'Not Detected Any Files From Path'
        
        rm_n_mkdir(self.output_dir + '/json/')
        rm_n_mkdir(self.output_dir + '/mat/')
        rm_n_mkdir(self.output_dir + '/overlay/')
        if self.save_qupath:
            rm_n_mkdir(self.output_dir + "/qupath/")

        # 20260125_GWJ_Test，不好权衡两个分类头的结果，效果不好
        # def proc_callback(results):
        #     """Post processing callback.
        #
        #     Output format is implicit assumption, taken from `_post_process_patches`
        #
        #     """
        #     img_name, pred_map, pred_inst, inst_info_dict, overlaid_img = results
        #
        #     nuc_val_list = list(inst_info_dict.values())
        #     # need singleton to make matlab happy
        #     nuc_uid_list = np.array(list(inst_info_dict.keys()))[:,None]
        #     nuc_type_list = np.array([v["type"] for v in nuc_val_list])[:,None]
        #     nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])
        #
        #     mat_dict = {
        #         "inst_map" : pred_inst,
        #         "inst_uid" : nuc_uid_list,
        #         "inst_type": nuc_type_list,
        #         "inst_centroid": nuc_coms_list
        #     }
        #     if self.nr_types is None: # matlab does not have None type array
        #         mat_dict.pop("inst_type", None)
        #
        #     if self.save_raw_map:
        #         mat_dict["raw_map"] = pred_map
        #     save_path = "%s/mat/%s.mat" % (self.output_dir, img_name)
        #     sio.savemat(save_path, mat_dict)
        #
        #     save_path = "%s/overlay/%s.png" % (self.output_dir, img_name)
        #     cv2.imwrite(save_path, cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR))
        #
        #     if self.save_qupath:
        #         nuc_val_list = list(inst_info_dict.values())
        #         nuc_type_list = np.array([v["type"] for v in nuc_val_list])
        #         nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])
        #         save_path = "%s/qupath/%s.tsv" % (self.output_dir, img_name)
        #         convert_format.to_qupath(
        #             save_path, nuc_coms_list, nuc_type_list, self.type_info_dict
        #         )
        #
        #     save_path = "%s/json/%s.json" % (self.output_dir, img_name)
        #     self.__save_json(save_path, inst_info_dict, None)
        #     return img_name
        #
        # 20260126_GWJ_test:分类头各自argmax,
        # def proc_callback(results):  # 无 self，修复报错
        #     """
        #     Post processing callback.
        #     Results: (img_name, pred_map, pred_inst, inst_info_dict, overlaid_img)
        #     """
        #     img_name, pred_map, pred_inst, inst_info_dict, overlaid_img = results
        #
        #     # =========================================================
        #     # 1. [核心逻辑] 保留背景判定 + 硬规则决策
        #     # =========================================================
        #     if pred_map.shape[-1] >= 10:
        #
        #         # 提取分类头的概率图
        #         map_t = pred_map[..., 4:7]  # TTF-1 (H, W, 3) [Bg, Neg, Pos]
        #         map_w = pred_map[..., 7:10]  # WT-1  (H, W, 3) [Bg, Neg, Pos]
        #
        #         # 遍历每一个检测到的细胞实例
        #         for uid, info in inst_info_dict.items():
        #             centroid = info['centroid']
        #             cy, cx = int(centroid[0]), int(centroid[1])
        #
        #             # 边界安全检查
        #             cy = min(cy, pred_map.shape[0] - 1)
        #             cx = min(cx, pred_map.shape[1] - 1)
        #
        #             # --- A. 获取每个头的独立分类结果 (Argmax 全通道) ---
        #             # 结果可能是: 0(Bg), 1(Neg), 2(Pos)
        #             prob_vec_t = map_t[cy, cx]
        #             prob_vec_w = map_w[cy, cx]
        #
        #             label_t = np.argmax(prob_vec_t[1:]) + 1
        #             label_w = np.argmax(prob_vec_w[1:]) + 1
        #
        #             # --- B. 执行硬性规则 ---
        #
        #             final_type = 3  # 默认为 Other
        #
        #             # 规则 1: 双背景 -> 背景 (0)
        #             # "如果TTF-1和WT-1都识别为类别0，则视为类别0"
        #             # if label_t == 0 and label_w == 0:
        #             #     final_type = 0
        #
        #             # 规则 2: TTF-1 阳性 (T=2, W=1) -> 腺癌 (1)
        #             if label_t == 2 and label_w == 1:
        #                 final_type = 1
        #
        #             # 规则 3: WT-1 阳性 (W=2, T=1) -> 间皮 (2)
        #             elif label_w == 2 and label_t == 1:
        #                 final_type = 2
        #
        #             # 规则 4: 双阴性 (T=1, W=1) -> 其他 (3)
        #             elif label_t == 1 and label_w == 1:
        #                 final_type = 3
        #
        #             # --- [兜底逻辑 / 冲突处理] ---
        #             # 处理规则未覆盖的边缘情况
        #             else:
        #                 # 情况 A: 双阳性冲突 (T=2, W=2) -> 比较概率
        #                 if label_t == 2 and label_w == 2:
        #                     if prob_vec_t[2] > prob_vec_w[2]:
        #                         final_type = 1
        #                     else:
        #                         final_type = 2
        #
        #                 # 情况 B: 单个阳性 (即使另一个头说是背景)
        #                 # 既然NP分支检测到了细胞，且有一个头说是阳性，我们倾向于相信阳性
        #                 elif label_t == 2:
        #                     final_type = 1
        #                 elif label_w == 2:
        #                     final_type = 2
        #
        #                 # 情况 C: 包含背景的阴性 (如 T=0, W=1)
        #                 # 倾向于判定为细胞(Other)，而不是背景
        #                 elif label_t == 1 or label_w == 1:
        #                     final_type = 3
        #
        #                 # 其他情况保持默认
        #
        #             # 更新字典
        #             inst_info_dict[uid]['type'] = int(final_type)
        #
        #             # (可选) 记录便于调试
        #             inst_info_dict[uid]['label_t'] = int(label_t)
        #             inst_info_dict[uid]['label_w'] = int(label_w)
        #
        #     # 打印分布
        #     # detected_types = [v["type"] for v in inst_info_dict.values()]
        #     # if len(detected_types) > 0:
        #     #     unique, counts = np.unique(detected_types, return_counts=True)
        #     #     print(f"[{img_name}] 最终分类统计: {dict(zip(unique, counts))}")
        #
        #     # =========================================================
        #     # 2. [可视化]
        #     # =========================================================
        #     repaint_dict = {
        #         0: ("nolabel", (0, 0, 0)),  # ⚫ 黑色/不显示
        #         1: ("Adeno", (255, 0, 0)),  # 🔴 红
        #         2: ("Meso", (0, 255, 0)),  # 🟢 绿
        #         3: ("Other", (0, 0, 255)),  # 🔵 蓝
        #     }
        #
        #     # 注意：使用 self.draw_dot
        #     new_colored_img = visualize_instances_dict(
        #         overlaid_img,
        #         inst_info_dict,
        #         draw_dot=self.draw_dot,
        #         type_colour=repaint_dict,
        #         line_thickness=2
        #     )
        #
        #     save_path = "%s/overlay/%s.png" % (self.output_dir, img_name)
        #     cv2.imwrite(save_path, cv2.cvtColor(new_colored_img, cv2.COLOR_RGB2BGR))
        #
        #     # =========================================================
        #     # 3. [保存数据] Mat / QuPath / JSON
        #     # =========================================================
        #     nuc_val_list = list(inst_info_dict.values())
        #     if len(nuc_val_list) > 0:
        #         nuc_uid_list = np.array(list(inst_info_dict.keys()))[:, None]
        #         nuc_type_list = np.array([v["type"] for v in nuc_val_list])[:, None]
        #         nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])
        #     else:
        #         nuc_uid_list = np.empty((0, 1))
        #         nuc_type_list = np.empty((0, 1))
        #         nuc_coms_list = np.empty((0, 2))
        #
        #     mat_dict = {
        #         "inst_map": pred_inst,
        #         "inst_uid": nuc_uid_list,
        #         "inst_type": nuc_type_list,
        #         "inst_centroid": nuc_coms_list
        #     }
        #     if self.save_raw_map:
        #         mat_dict["raw_map"] = pred_map
        #     sio.savemat("%s/mat/%s.mat" % (self.output_dir, img_name), mat_dict)
        #
        #     if self.save_qupath:
        #         convert_format.to_qupath(
        #             "%s/qupath/%s.tsv" % (self.output_dir, img_name),
        #             nuc_coms_list, nuc_type_list.flatten(), repaint_dict
        #         )
        #
        #     self.__save_json("%s/json/%s.json" % (self.output_dir, img_name), inst_info_dict, None)
        #
        #     return img_name

        # 20260204_GWJ,基于全核平均概率的高性能后处理
            from scipy.ndimage import mean as nd_mean  # 确保在顶部或此处引入

            def proc_callback(results):  # 无 self，修复报错
                """
                Post processing callback.
                Results: (img_name, pred_map, pred_inst, inst_info_dict, overlaid_img)
                """
                img_name, pred_map, pred_inst, inst_info_dict, overlaid_img = results

                # =========================================================
                # 1. [核心逻辑] 基于全核区域平均概率的硬规则决策
                # =========================================================
                if pred_map.shape[-1] >= 10 and len(inst_info_dict) > 0:

                    # 提取分类头的概率图
                    # map_t: TTF-1 (H, W, 3) -> [Bg, Neg, Pos]
                    # map_w: WT-1  (H, W, 3) -> [Bg, Neg, Pos]
                    map_t = pred_map[..., 4:7]
                    map_w = pred_map[..., 7:10]

                    # 获取所有实例 ID 列表
                    uid_list = list(inst_info_dict.keys())

                    # --- 高性能批量计算：一次性获取所有实例在各个通道的平均概率 ---
                    # 使用 scipy.ndimage.mean 避免了 O(N) 的循环 mask 操作
                    # 返回值形状均为 (len(uid_list),)
                    def get_avg_probs(prob_map, mask, index_list):
                        # 分别计算每个通道的均值
                        c0 = nd_mean(prob_map[..., 0], labels=mask, index=index_list)
                        c1 = nd_mean(prob_map[..., 1], labels=mask, index=index_list)
                        c2 = nd_mean(prob_map[..., 2], labels=mask, index=index_list)
                        return np.stack([c0, c1, c2], axis=-1)

                    avg_probs_t = get_avg_probs(map_t, pred_inst, uid_list)
                    avg_probs_w = get_avg_probs(map_w, pred_inst, uid_list)

                    # 遍历计算结果并应用规则
                    for idx, uid in enumerate(uid_list):
                        prob_vec_t = avg_probs_t[idx]
                        prob_vec_w = avg_probs_w[idx]

                        # --- A. 获取每个头的独立分类结果 (排除背景通道 0，在 Neg 和 Pos 中 argmax) ---
                        # 结果可能是: 1(Neg), 2(Pos)
                        label_t = np.argmax(prob_vec_t[1:]) + 1
                        label_w = np.argmax(prob_vec_w[1:]) + 1

                        # --- B. 执行硬性规则 ---
                        final_type = 3  # 默认为 Other

                        # 规则 1: 背景(0)
                        if label_t == 0 and label_w == 0:
                            final_type = 0
                        # 规则 2: TTF-1 阳性 (T=2, W=1) -> 腺癌 (1)
                        elif label_t == 2 and label_w == 1:
                            final_type = 1

                        # 规则 3: WT-1 阳性 (W=2, T=1) -> 间皮 (2)
                        elif label_w == 2 and label_t == 1:
                            final_type = 2

                        # 规则 4: 双阴性 (T=1, W=1) -> 其他 (3)
                        elif label_t == 1 and label_w == 1:
                            final_type = 3

                        # --- [兜底逻辑 / 冲突处理] ---
                        else:
                            # 情况 A: 双阳性冲突 (T=2, W=2) -> 比较 Pos 通道的平均概率
                            if label_t == 2 and label_w == 2:
                                if prob_vec_t[2] > prob_vec_w[2]:
                                    final_type = 1
                                else:
                                    final_type = 2

                            # 情况 B: 单个阳性 (即使另一个头偏向背景)
                            elif label_t == 2:
                                final_type = 1
                            elif label_w == 2:
                                final_type = 2

                            # 情况 C: 包含背景的阴性 (如 T=Bg, W=Neg)
                            elif label_t == 1 or label_w == 1:
                                final_type = 3

                        # 更新字典
                        inst_info_dict[uid]['type'] = int(final_type)
                        # 记录平均概率供后期分析
                        inst_info_dict[uid]['avg_prob_t'] = prob_vec_t.tolist()
                        inst_info_dict[uid]['avg_prob_w'] = prob_vec_w.tolist()

                # =========================================================
                # 2. [可视化]
                # =========================================================
                repaint_dict = {
                    0: ("nolabel", (0, 0, 0)),  # ⚫ 黑色/不显示
                    1: ("Adeno", (255, 0, 0)),  # 🔴 红
                    2: ("Meso", (0, 255, 0)),  # 🟢 绿
                    3: ("Other", (0, 0, 255)),  # 🔵 蓝
                }

                # 注意：使用 self.draw_dot
                new_colored_img = visualize_instances_dict(
                    overlaid_img,
                    inst_info_dict,
                    draw_dot=self.draw_dot,
                    type_colour=repaint_dict,
                    line_thickness=2
                )

                save_path = "%s/overlay/%s.png" % (self.output_dir, img_name)
                cv2.imwrite(save_path, cv2.cvtColor(new_colored_img, cv2.COLOR_RGB2BGR))

                # =========================================================
                # 3. [保存数据] Mat / QuPath / JSON
                # =========================================================
                nuc_val_list = list(inst_info_dict.values())
                if len(nuc_val_list) > 0:
                    nuc_uid_list = np.array(list(inst_info_dict.keys()))[:, None]
                    nuc_type_list = np.array([v["type"] for v in nuc_val_list])[:, None]
                    nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])
                else:
                    nuc_uid_list = np.empty((0, 1))
                    nuc_type_list = np.empty((0, 1))
                    nuc_coms_list = np.empty((0, 2))

                mat_dict = {
                    "inst_map": pred_inst,
                    "inst_uid": nuc_uid_list,
                    "inst_type": nuc_type_list,
                    "inst_centroid": nuc_coms_list
                }
                if self.save_raw_map:
                    mat_dict["raw_map"] = pred_map

                # 保存 .mat 文件
                sio.savemat("%s/mat/%s.mat" % (self.output_dir, img_name), mat_dict)

                # 保存 QuPath 兼容格式
                if self.save_qupath:
                    convert_format.to_qupath(
                        "%s/qupath/%s.tsv" % (self.output_dir, img_name),
                        nuc_coms_list, nuc_type_list.flatten(), repaint_dict
                    )

                # 保存 JSON 结果
                self.__save_json("%s/json/%s.json" % (self.output_dir, img_name), inst_info_dict, None)

                return img_name

        def detach_items_of_uid(items_list, uid, nr_expected_items):
            item_counter = 0
            detached_items_list = []
            remained_items_list = []
            while True:
                pinfo, pdata = items_list.pop(0)
                pinfo = np.squeeze(pinfo)
                if pinfo[-1] == uid:
                    detached_items_list.append([pinfo, pdata])
                    item_counter += 1
                else:
                    remained_items_list.append([pinfo, pdata])
                if item_counter == nr_expected_items:
                    break
            # do this to ensure the ordering
            remained_items_list = remained_items_list + items_list
            return detached_items_list, remained_items_list

        proc_pool = None
        if self.nr_post_proc_workers > 0:
            proc_pool = ProcessPoolExecutor(self.nr_post_proc_workers)

        while len(file_path_list) > 0:

            hardware_stats = psutil.virtual_memory()
            available_ram = getattr(hardware_stats, "available")
            available_ram = int(available_ram * self.mem_usage)
            # available_ram >> 20 for MB, >> 30 for GB

            # TODO: this portion looks clunky but seems hard to detach into separate func

            # * caching N-files into memory such that their expected (total) memory usage
            # * does not exceed the designated percentage of currently available memory
            # * the expected memory is a factor w.r.t original input file size and
            # * must be manually provided
            file_idx = 0
            use_path_list = []
            cache_image_list = []
            cache_patch_info_list = []
            cache_image_info_list = []
            while len(file_path_list) > 0:
                file_path = file_path_list.pop(0)

                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                src_shape = img.shape

                img, patch_info, top_corner = _prepare_patching(
                    img, self.patch_input_shape, self.patch_output_shape, True
                )
                self_idx = np.full(patch_info.shape[0], file_idx, dtype=np.int32)
                patch_info = np.concatenate([patch_info, self_idx[:, None]], axis=-1)
                # ? may be expensive op
                patch_info = np.split(patch_info, patch_info.shape[0], axis=0)
                patch_info = [np.squeeze(p) for p in patch_info]

                # * this factor=5 is only applicable for HoVerNet
                expected_usage = sys.getsizeof(img) * 5
                available_ram -= expected_usage
                if available_ram < 0:
                    break

                file_idx += 1
                # if file_idx == 4: break
                use_path_list.append(file_path)
                cache_image_list.append(img)
                cache_patch_info_list.extend(patch_info)
                # TODO: refactor to explicit protocol
                cache_image_info_list.append([src_shape, len(patch_info), top_corner])

            # * apply neural net on cached data
            dataset = SerializeFileList(
                cache_image_list, cache_patch_info_list, self.patch_input_shape
            )

            dataloader = data.DataLoader(
                dataset,
                num_workers=self.nr_inference_workers,
                batch_size=self.batch_size,
                drop_last=False,
            )

            pbar = tqdm.tqdm(
                desc="Process Patches",
                leave=True,
                total=int(len(cache_patch_info_list) / self.batch_size) + 1,
                ncols=80,
                ascii=True,
                position=0,
            )

            accumulated_patch_output = []
            for batch_idx, batch_data in enumerate(dataloader):
                sample_data_list, sample_info_list = batch_data
                sample_output_list = self.run_step(sample_data_list)
                sample_info_list = sample_info_list.numpy()
                curr_batch_size = sample_output_list.shape[0]
                sample_output_list = np.split(
                    sample_output_list, curr_batch_size, axis=0
                )
                sample_info_list = np.split(sample_info_list, curr_batch_size, axis=0)
                sample_output_list = list(zip(sample_info_list, sample_output_list))
                accumulated_patch_output.extend(sample_output_list)
                pbar.update()
            pbar.close()

            # * parallely assemble the processed cache data for each file if possible
            future_list = []
            for file_idx, file_path in enumerate(use_path_list):
                image_info = cache_image_info_list[file_idx]
                file_ouput_data, accumulated_patch_output = detach_items_of_uid(
                    accumulated_patch_output, file_idx, image_info[1]
                )

                # * detach this into func and multiproc dispatch it
                src_pos = image_info[2]  # src top left corner within padded image
                src_image = cache_image_list[file_idx]
                src_image = src_image[
                    src_pos[0] : src_pos[0] + image_info[0][0],
                    src_pos[1] : src_pos[1] + image_info[0][1],
                ]

                base_name = pathlib.Path(file_path).stem
                file_info = {
                    "src_shape": image_info[0],
                    "src_image": src_image,
                    "name": base_name,
                }

                post_proc_kwargs = {
                    # "nr_types": self.nr_types,
                    # 20260123_GWJ
                    "nr_types": 4,
                    "return_centroids": True,
                }  # dynamicalize this

                overlay_kwargs = {
                    "draw_dot": self.draw_dot,
                    "type_colour": self.type_info_dict,
                    "line_thickness": 2,
                }
                func_args = (
                    self.post_proc_func,
                    post_proc_kwargs,
                    file_ouput_data,
                    file_info,
                    overlay_kwargs,
                )

                # dispatch for parallel post-processing
                if proc_pool is not None:
                    proc_future = proc_pool.submit(_post_process_patches, *func_args)
                    # ! manually poll future and call callback later as there is no guarantee
                    # ! that the callback is called from main thread
                    future_list.append(proc_future)
                else:
                    proc_output = _post_process_patches(*func_args)
                    proc_callback(proc_output)

            if proc_pool is not None:
                # loop over all to check state a.k.a polling
                for future in as_completed(future_list):
                    # TODO: way to retrieve which file crashed ?
                    # ! silent crash, cancel all and raise error
                    if future.exception() is not None:
                        log_info("Silent Crash")
                        # ! cancel somehow leads to cascade error later
                        # ! so just poll it then crash once all future
                        # ! acquired for now
                        # for future in future_list:
                        #     future.cancel()
                        # break
                    else:
                        file_path = proc_callback(future.result())
                        log_info("Done Assembling %s" % file_path)
        return

