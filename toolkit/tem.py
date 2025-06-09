import os
import shutil

# 定义源文件夹和目标文件夹
source_folder = r'E:\护栏中心高度总文件夹\成雅高速_原始数据\将数据合到一起\labels'  # 替换为包含 txt 文件的文件夹路径
target_folder = r'E:\护栏中心高度总文件夹\成雅高速_原始数据\将数据合到一起'  # 替换为目标文件夹路径

# 创建 0, 1, 2, 3 文件夹
for i in range(4):
    os.makedirs(os.path.join(target_folder, str(i)), exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith('.txt'):
        # 去掉扩展名
        base_name = os.path.splitext(filename)[0]

        # 构造 JPEG 图片的路径
        jpeg_image_path = os.path.join(source_folder[:-6], base_name + '.jpeg')

        # 读取 txt 文件的第一行
        with open(os.path.join(source_folder, filename), 'r') as file:
            first_line = file.readline().strip()

        # 获取最后一个字符
        if first_line:
            last_char = first_line[-1]

            # 检查最后一个字符是否在 0, 1, 2, 3 中
            if last_char in '0123':
                # 构造目标路径
                target_path = os.path.join(target_folder, last_char, base_name + '.jpeg')

                # 复制 JPEG 图片到目标文件夹
                try:
                    shutil.copy(jpeg_image_path, target_path)
                    print(f'Copied {jpeg_image_path} to {target_path}')
                except FileNotFoundError:
                    print(f'Image {jpeg_image_path} does not exist.')
                except Exception as e:
                    print(f'Error copying {jpeg_image_path} to {target_path}: {e}')