import json
import cv2
import numpy as np
import os
from pathlib import Path


def batch_draw_and_save(image_folder, json_folder, output_folder):
    """
    针对配准图像的批量绘制：将 HE target 的标注结果画在对应的 Warped Source IHC 图像上
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    json_files = list(Path(json_folder).glob("*.json"))
    print(f"找到 {len(json_files)} 个 JSON 文件，准备开始处理...")

    for json_path in json_files:
        # 1. 文件名映射逻辑
        raw_json_name = json_path.stem

        # 映射规则：
        # -1_HE_target -> -1_TTF-1_warped_source
        # -2_HE_target -> -2_WT-1_warped_source
        if "-1_HE_target" in raw_json_name:
            target_img_name = raw_json_name.replace("-1_HE_target", "-1_TTF-1_warped_source")
        elif "-2_HE_target" in raw_json_name:
            target_img_name = raw_json_name.replace("-2_HE_target", "-2_WT-1_warped_source")
        else:
            # 如果不符合上述命名规则，尝试按原名匹配
            target_img_name = raw_json_name

        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            possible_path = Path(image_folder) / (target_img_name + ext)
            if possible_path.exists():
                img_path = possible_path
                break

        if img_path is None:
            print(f"跳过: 映射后的图片名 {target_img_name} 在库中未找到")
            continue

        # 2. 读取图片和数据
        img = cv2.imread(str(img_path))
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 3. 绘图配置 (根据 type 分类)
        # BGR格式：1-腺癌(红), 2-间皮(绿), 3-其它(蓝)
        color_map = {0: (0, 0, 0),
                     1: (0, 0, 255),  # Red
                     2: (0, 255, 0),  # Green
                     3: (255, 0, 0),  # Blue
                     }

        for nuc_id, info in data["nuc"].items():
            # 转换轮廓坐标
            contour = np.array(info["contour"], dtype=np.int32).reshape((-1, 1, 2))

            # 获取颜色
            nuc_type = info.get("type", 1)
            color = color_map.get(nuc_type, (255, 255, 255))

            # 绘制轮廓 (线宽建议设为1或2，医学图像建议细一点以便观察染色)
            cv2.drawContours(img, [contour], -1, color, 2)

        # 4. 保存结果 (文件名包含原始信息，方便追溯)
        output_path = os.path.join(output_folder, f"{target_img_name}.png")
        cv2.imwrite(output_path, img)
        print(f"已完成: {json_path.name} -> {target_img_name}")

    print("--- 所有任务处理完毕 ---")




    # 改为可以选择模式
# --- 路径配置 (保持你提供的路径) ---
IMG_DIR = "/home/wenjun_data/SMILE_Data/Data/20251218-1DeeperHistReg/infer_test/B24-02371-1_HE_target_2048x2048"
JSON_DIR = "/home/wenjun_data/SMILE_Data/Data/20251218-1DeeperHistReg/infer_test/Result/HoVerNet/20260125-3_B24-02371-1/json"
SAVE_DIR = "/home/wenjun_data/SMILE_Data/Data/20251218-1DeeperHistReg/infer_test/Result/HoVerNet/20260125-3_B24-02371-1/HE_overlay"

batch_draw_and_save(IMG_DIR, JSON_DIR, SAVE_DIR)