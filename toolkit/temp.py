import os
import shutil
import random

# 定义文件夹路径
source_folder = r'E:\护栏中心高度总文件夹\训练集\审核12万张\0'  # 替换为源文件夹的路径
destination_folder = r'E:\护栏中心高度总文件夹\训练集\审核12万张\0_训练集'  # 替换为目标文件夹的路径

# 创建目标文件夹（如果不存在）
os.makedirs(destination_folder, exist_ok=True)

# 获取源文件夹中的所有图片文件
all_images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# 随机挑选1000张图片，如果图片数量不足1000张，则挑选所有图片
num_images_to_copy = min(6000, len(all_images))
selected_images = random.sample(all_images, num_images_to_copy)

# 复制选中的图片到目标文件夹
for image in selected_images:
    source_path = os.path.join(source_folder, image)
    destination_path = os.path.join(destination_folder, image)
    shutil.copy2(source_path, destination_path)
    print(f"Copied: {image} to {destination_folder}")

print("随机图片复制完成！")
# import os
# import shutil
#
# # 定义文件夹路径
# folder1 = r'D:\WJ\Pycharm_workspace\ultralytics_guard_bar\runs\classify\predict9\labels'  # 替换为实际的文件夹1路径
# folder2 = r'E:\护栏中心高度总文件夹\成雅高速_原始数据\将数据合到一起_挑选'  # 替换为实际的文件夹2路径
# folder3 = r'E:\护栏中心高度总文件夹\成雅高速_原始数据\分类'  # 替换为实际的文件夹3路径
#
# # 确保文件夹3存在
# os.makedirs(folder3, exist_ok=True)
#
# # 创建子文件夹 0, 1, 2, 3
# subfolders = [os.path.join(folder3, str(i)) for i in range(4)]
# for subfolder in subfolders:
#     os.makedirs(subfolder, exist_ok=True)
#
# # 遍历文件夹1中的所有txt文件
# for filename in os.listdir(folder1):
#     if filename.endswith('.txt'):
#         txt_file_path = os.path.join(folder1, filename)
#
#         # 读取txt文件的第一行
#         with open(txt_file_path, 'r') as file:
#             first_line = file.readline().strip()
#             last_number = int(first_line.split()[-1])  # 获取最后一个数字
#
#         # 去掉扩展名
#         base_name = os.path.splitext(filename)[0]
#
#         # 生成文件夹2中对应的图片路径
#         image_path = os.path.join(folder2, base_name)  # 图片文件名去掉扩展名
#
#         # 检查对应的图片文件是否存在，并复制到相应的子文件夹
#         if os.path.isfile(image_path + '.jpeg'):  # 假设图片为.jpg格式
#             shutil.copy(image_path + '.jpeg', subfolders[last_number])
#         elif os.path.isfile(image_path + '.png'):  # 如果是.png格式
#             shutil.copy(image_path + '.png', subfolders[last_number])
#         else:
#             print(f"Warning: Image for {filename} not found.")
#
# print("文件复制完成！")