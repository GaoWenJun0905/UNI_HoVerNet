import numpy as np
import glob
import os
import tqdm
from collections import Counter


def analyze_dataset_instances(data_dir):
    """
    遍历生成的patch，统计实例总数和类别分布
    """
    # 获取目录下所有的 .npy 文件
    file_list = glob.glob(os.path.join(data_dir, "*.npy"))
    file_list.sort()

    total_patches = len(file_list)
    if total_patches == 0:
        print(f"错误：在路径 {data_dir} 下没有找到 .npy 文件。")
        return

    print(f"正在分析路径: {data_dir}")
    print(f"找到 {total_patches} 个 patches，开始遍历...")

    all_type_counts = Counter()
    total_instances = 0

    # 使用 tqdm 显示进度
    for path in tqdm.tqdm(file_list, ascii=True, desc="Analyzing"):
        # load 数据 (H, W, 5)
        patch_data = np.load(path)

        # 提取 instance 矩阵 (index 3) 和 type 矩阵 (index 4)
        inst_map = patch_data[..., 3].astype(np.int32)
        type_map = patch_data[..., 4].astype(np.int32)

        # 1. 统计当前 patch 的实例 ID（排除背景 0）
        # unique_insts = np.unique(inst_map)
        # unique_insts = unique_insts[unique_insts > 0]
        # num_insts = len(unique_insts)
        # total_instances += num_insts

        # 2. 更高级的统计：按类别统计实例
        # 找到每个实例对应的类别
        # 我们取每个实例在 type_map 中对应位置出现最多的值作为该实例的类别
        unique_insts = np.unique(inst_map)
        unique_insts = unique_insts[unique_insts > 0]

        for inst_id in unique_insts:
            total_instances += 1
            # 找到该实例在 type_map 对应的像素点
            pixels_of_this_inst = type_map[inst_map == inst_id]
            # 取众数作为该实例的类别（防止标注边缘微小误差）
            inst_type = Counter(pixels_of_this_inst).most_common(1)[0][0]
            all_type_counts[inst_type] += 1

    # ========================================================
    # 输出结果
    # ========================================================
    print("\n" + "=" * 30)
    print(f"数据集分析报告")
    print("-" * 30)
    print(f"Patch 总数:    {total_patches}")
    print(f"实例总数:      {total_instances}")
    print("-" * 30)
    print("各类别实例分布:")
    # 按 Type ID 排序输出
    for type_id in sorted(all_type_counts.keys()):
        count = all_type_counts[type_id]
        percentage = (count / total_instances) * 100 if total_instances > 0 else 0
        print(f"  Type {type_id}: {count:>8} 个 ({percentage:>6.2f}%)")
    print("=" * 30)


if __name__ == "__main__":
    # 替换为你实际生成的 train 或 valid 目录

    DATA_PATH = "/home/wenjun_data/SMILE_Data/Data/20260203_1229TTF-1/TrainingData/0.9:0.1_TTF-1_sldie4/f1/f1/train/540x540_500x500"

    analyze_dataset_instances(DATA_PATH)