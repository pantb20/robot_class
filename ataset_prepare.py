import os
import shutil
import random

# ================= 1. 参数配置 =================
# 原始数据集所在的根目录路径（如果脚本就在该目录下，填 '.' 即可）
source_dir = '.' 

# 整理后输出的目标根目录
target_dir = './dataset_split'

# 验证集比例（0.2 代表 80% 训练集，20% 验证集）
val_ratio = 0.2

# 随机种子，保证每次运行划分结果一致
random.seed(42)

# ================= 2. 创建目标文件夹 =================
for split in ['train', 'val']:
    for label in ['on', 'off']:
        os.makedirs(os.path.join(target_dir, split, label), exist_ok=True)

# ================= 3. 收集开灯和关灯的所有样本文件夹 =================
all_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f)) and f != 'dataset_split']

on_folders = []
off_folders = []

# 根据文件夹命名特征区分开灯与关灯
for folder in all_folders:
    if folder.startswith('2026'):  # 带时间戳的前缀为关灯数据集
        off_folders.append(folder)
    elif folder.isdigit():          # 纯数字短名称为开灯数据集
        on_folders.append(folder)

print(f"统计到的样本数 -> 开灯(on): {len(on_folders)} 个, 关灯(off): {len(off_folders)} 个")

# ================= 4. 定义处理与拷贝函数 =================
def process_category(folder_list, label_name):
    # 随机打乱样本顺序
    random.shuffle(folder_list)
    
    # 计算验证集数量
    val_count = int(len(folder_list) * val_ratio)
    val_folders = folder_list[:val_count]
    train_folders = folder_list[val_count:]
    
    # 复制文件的内部函数
    def copy_files(folders, split_name):
        for idx, folder in enumerate(folders):
            folder_path = os.path.join(source_dir, folder)
            
            # 寻找文件夹内部包含 raw 的图片 (兼容 raw.jpg, raw.png 等)
            raw_img_name = None
            for file in os.listdir(folder_path):
                if 'raw' in file.lower() and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    raw_img_name = file
                    break
            
            if raw_img_name:
                src_path = os.path.join(folder_path, raw_img_name)
                # 使用文件夹名重新命名图片，避免重复覆盖 (例如: on_000005.jpg)
                ext = os.path.splitext(raw_img_name)[1]
                dst_name = f"{label_name}_{folder}{ext}"
                dst_path = os.path.join(target_dir, split_name, label_name, dst_name)
                
                shutil.copy(src_path, dst_path)
            else:
                print(f"警告: 文件夹 {folder} 中未找到包含 'raw' 的图片！")

    copy_files(train_folders, 'train')
    copy_files(val_folders, 'val')
    print(f"[{label_name}] 处理完成: 训练集 {len(train_folders)} 张, 验证集 {len(val_folders)} 张")

# ================= 5. 执行划分 =================
process_category(on_folders, 'on')
process_category(off_folders, 'off')

print(f"\n数据集划分成功！已整理至目录: {os.path.abspath(target_dir)}")