from ultralytics import YOLO
import time
import os
from PIL import Image
# Load a model
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    # # Load a model
    # model = YOLO("runs/classify/train13/weights/best.pt")  # load a custom trained model
    # # Export the model
    # model.export(format="engine", imgsz=640, batch=100, dynamic=True)

    start_time = time.time()
    # Load a model
    model = YOLO(r"D:\WJ\Pycharm_workspace\ultralytics_guard_bar\runs\classify\train8\weights\best.pt")  # load a custom model

    # Predict with the model
    directory = r'E:\护栏中心高度总文件夹\训练集\guard_bar_new\train\2'  # 替换为您的目录路径

    # 初始化一个空列表来存储图片
    images = []

    # 获取指定目录下的所有文件，并排序
    files = sorted(os.listdir(directory))

    # for file in files:
    #     file_path = os.path.join(directory, file)
    #
    #     # 检查文件是否是图片（可以根据需要修改支持的扩展名）
    #     if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
    #         try:
    #             # 打开图片并添加到列表中
    #             image = Image.open(file_path)
    #             images.append(image)
    #             print(f"已加载图片: {file}")
    #         except Exception as e:
    #             print(f"无法打开图片 {file}: {e}")

    image_folder = r'E:\护栏中心高度总文件夹\成雅高速_原始数据\将数据合到一起'
    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpeg', '.png'))]

    batch_size = 100  # 设置合适的批次大小
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        results = model(batch_images, save=True, batch=len(batch_images), save_txt=True)
    # results = model(r'E:\护栏中心高度总文件夹\成雅高速_原始数据\将数据合到一起', save=True, batch=1, save_txt=True)
    print(results[0].probs.top1)
    # predict on an image
    end_time = time.time()  # 获取结束时间（秒）
    # 计算执行时间并转换为毫秒
    execution_time_ms = (end_time - start_time) * 1000
    print(f"执行时间: {execution_time_ms:.3f} 毫秒")