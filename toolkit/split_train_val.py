import os
import random
import shutil

# 定义源文件夹和目标文件夹
source_folder = r'E:\护栏中心高度总文件夹\训练集\guard_bar_20250520\3'  # 替换为您的源文件夹路径
target_folder_2 = r'E:\护栏中心高度总文件夹\训练集\guard_bar_20250520\train\3'  # 替换为目标文件夹2的路径
target_folder_3 = r'E:\护栏中心高度总文件夹\训练集\guard_bar_20250520\val\3'  # 替换为目标文件夹3的路径

# 创建目标文件夹（如果不存在）
os.makedirs(target_folder_2, exist_ok=True)
os.makedirs(target_folder_3, exist_ok=True)

# 获取源文件夹中的所有图片文件
images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# 打乱文件顺序
random.shuffle(images)

# 计算分割点
split_index = int(len(images) * 0.8)

# 将前 80% 的图片复制到文件夹2，后 20% 的复制到文件夹3
for i in range(len(images)):
    source_path = os.path.join(source_folder, images[i])
    if i < split_index:
        target_path = os.path.join(target_folder_2, images[i])
    else:
        target_path = os.path.join(target_folder_3, images[i])

    # 复制文件
    shutil.copy(source_path, target_path)

print(f'Copied {split_index} images to {target_folder_2} and {len(images) - split_index} images to {target_folder_3}.')