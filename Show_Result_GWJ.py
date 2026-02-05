import os
import glob
import re
import random
from PIL import Image, ImageDraw, ImageFont

# ================= 核心配置区域 =================

# 1. 路径设置
HE_DIR = r'/home/wenjun_data/SMILE_Data/Data/20251218-1DeeperHistReg/infer_test/Result/HoVerNet/20260204-1_B24-02371-1/overlay'
IHC_DIR = r'/home/wenjun_data/SMILE_Data/Data/20251218-1DeeperHistReg/infer_test/B24-02371-1_TTF-1_warped_source_2048x2048'
SAVE_DIR = r'/home/wenjun_data/SMILE_Data/Data/20251218-1DeeperHistReg/infer_test/Result/HoVerNet/20260204-1_B24-02371-1/overlay/Visualization'

# 2. 模式选择 (修改这里！)
# 选项: 'random' (随机抽取)  或  'specific' (指定具体坐标)
# MODE = 'random'
MODE = 'specific'

# --- 模式 A: 如果选择了 'random' ---
RANDOM_NUM = 30  # 随机抽多少张

# --- 模式 B: 如果选择了 'specific' ---
# 在这里填入你想查看的坐标 (即文件名中 _HE_target_ 后面的那串数字)
SPECIFIC_LIST = [
    '1027_31519',
    # '11267_6943',
    # 可以继续添加...
]

# ================= 样式配置 =================
STYLE = {
    'bg_color': (240, 240, 240), 'text_color': (50, 50, 50),
    'border_color': (0, 0, 0), 'padding': 30,
    'gap': 20, 'header_height': 80,
    'sub_header_height': 40,
}


# ===========================================

def create_styled_image(img_he, img_ihc, title_text, sub_he, sub_ihc):
    """ 创建美观对比图 (保持不变) """
    if img_he.size != img_ihc.size:
        img_ihc = img_ihc.resize(img_he.size, Image.LANCZOS)
    w, h = img_he.size
    total_w = STYLE['padding'] * 2 + w * 2 + STYLE['gap']
    total_h = STYLE['header_height'] + STYLE['sub_header_height'] + h + STYLE['padding']

    canvas = Image.new('RGB', (total_w, total_h), STYLE['bg_color'])
    draw = ImageDraw.Draw(canvas)

    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font_title = ImageFont.truetype(font_path, 28)
        font_sub = ImageFont.truetype(font_path, 20)
    except:
        font_title = ImageFont.load_default();
        font_sub = ImageFont.load_default()

    # 绘制文字
    draw.text((total_w // 2, STYLE['header_height'] // 2), title_text, fill=STYLE['text_color'], font=font_title,
              anchor="mm")

    # 粘贴图片
    x1, y = STYLE['padding'], STYLE['header_height'] + STYLE['sub_header_height']
    x2 = x1 + w + STYLE['gap']
    canvas.paste(img_he, (x1, y));
    canvas.paste(img_ihc, (x2, y))

    # 边框
    draw.rectangle([x1 - 1, y - 1, x1 + w, y + h], outline=STYLE['border_color'])
    draw.rectangle([x2 - 1, y - 1, x2 + w, y + h], outline=STYLE['border_color'])

    # 子标题
    sub_y = STYLE['header_height'] + STYLE['sub_header_height'] // 2
    draw.text((x1 + w // 2, sub_y), sub_he, fill=STYLE['text_color'], font=font_sub, anchor="mm")
    draw.text((x2 + w // 2, sub_y), sub_ihc, fill=(0, 50, 150), font=font_sub, anchor="mm")

    return canvas


def run_visualization():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    # 1. 扫描文件
    he_pattern = os.path.join(HE_DIR, "*_HE_target_*")
    he_files = [f for f in sorted(glob.glob(he_pattern)) if f.lower().endswith(('.png', '.tif', '.jpg'))]

    if not he_files:
        print("未找到HE文件。");
        return

    print(f"正在扫描匹配... (模式: {MODE})")
    regex_he = re.compile(r"^(.*?)_HE_target_(.*)\.(png|tif|tiff|jpg)$", re.IGNORECASE)
    matched_pairs = []

    for he_path in he_files:
        match = regex_he.match(os.path.basename(he_path))
        if match:
            pid, coords = match.group(1), match.group(2)
            ihc_candidates = glob.glob(os.path.join(IHC_DIR, f"{pid}_*_warped_source_{coords}.*"))
            if ihc_candidates:
                matched_pairs.append({'he': he_path, 'ihc': ihc_candidates[0], 'id': pid, 'coords': coords})

    total = len(matched_pairs)
    print(f"共找到 {total} 对匹配图片。")
    if total == 0: return

    # 2. 根据模式筛选
    final_selection = []

    if MODE == 'specific':
        # 指定模式
        target_set = set(SPECIFIC_LIST)
        print(f"正在查找指定的 {len(target_set)} 张图片: {target_set}")

        for item in matched_pairs:
            if item['coords'] in target_set:
                final_selection.append(item)
                target_set.discard(item['coords'])

        if target_set:
            print(f"警告: 以下坐标未找到对应文件: {target_set}")

    elif MODE == 'random':
        # 随机模式
        count = min(total, RANDOM_NUM)
        print(f"正在随机抽取 {count} 张...")
        final_selection = random.sample(matched_pairs, count)

    else:
        print("错误: MODE 配置不正确，请设置为 'random' 或 'specific'")
        return

    # 3. 生成并保存
    print(f"开始生成 {len(final_selection)} 张对比图...\n")
    for i, item in enumerate(final_selection):
        try:
            img_he = Image.open(item['he']).convert("RGB")
            img_ihc = Image.open(item['ihc']).convert("RGB")

            # 提取染色名
            ihc_fname = os.path.basename(item['ihc'])
            try:
                stain = ihc_fname.replace(item['id'] + "_", "").split("_warped")[0]
            except:
                stain = "IHC"

            # 绘图
            canvas = create_styled_image(
                img_he, img_ihc,
                f"Case: {item['id']} | Coords: {item['coords']}",
                "H&E (Target)", f"IHC ({stain}) (Warped)"
            )

            save_name = f"Compare_{i + 1:02d}_{item['id']}_{item['coords']}.jpg"
            canvas.save(os.path.join(SAVE_DIR, save_name), quality=95)
            print(f"[{i + 1}/{len(final_selection)}] 已保存: {save_name}")

        except Exception as e:
            print(f"处理出错: {e}")

    print(f"\n全部完成！结果保存在: {SAVE_DIR}")


if __name__ == "__main__":
    run_visualization()