import json
from pycocotools import mask as mask_utils  # type: ignore
import pycocotools
import numpy as np
import cv2
import os
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt

def frPyObjects(directory_path):
    id_class = [  "pole", "light", "trafic_sign", "road_edge",
              "road_objects", "anti-capture", "speed_bump", "height_limiting_device", "width_limiting_device",
              "service_area", "parking_area", "passenger_station", "management_device", "afforest_protection_device",
              "school", "hospital", "Isolation_barrier", "Anti_glare_board", "gas_station",
              "Water_horse", "Cone_bucket", "Sound_barrier", "background", "guard_bar", "concrete_guardrail",
              "cable_guardrail", "side_fence_guardrail"]
    colors = {"background": (107, 167, 213), "concrete_guardrail": (0, 255, 0), 'gas_station': (0, 0, 255),
              'guard_bar':(0, 255, 255) ,'lane': (255, 255, 0), 'light': (255, 0, 255),
              'pillar': (255, 165, 0), 'pole': (128, 0, 128), 'road': (0, 100, 0),
              'road_edge': (173, 216, 230), 'side_fence_guardrail': (211, 211, 211),
              'sign': (0, 0, 200), 'Sound_barrier': (255, 215, 0), 'trafic_sign': (255, 192, 203),"oad_edge": (173, 215, 203)}

    # color_mask = colors[ann['label']] #指定生成一种与标签对应的颜色

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        file_name_without_extension, file_extension = os.path.splitext(filename)
        img_path = os.path.join(directory_path.replace('annotations', 'images'), file_name_without_extension + '.jpg')
        with open(file_path, 'r', encoding='utf-8') as file:
            json_str = json.load(file)
        # 解析 JSON 数据
        original_image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        # image = Image.fromarray(original_image)
        merged_image = np.zeros((1200, 1920, 3), dtype=np.uint8)
        for str in json_str:
            counts = str['segmentation']['counts']
            category_id = str['category_id']
            class_id = id_class[category_id]
            color = colors[class_id]

            decoded_mask = pycocotools.mask.decode(str['segmentation'])
            y_coords, x_coords = np.where(decoded_mask == 1)#x和y的坐标，接下注释，然后接着写
            if len(y_coords) > 0 and len(x_coords) > 0:  # 确保掩码非空
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
            if class_id == 'background':
                alpha = 0.35
            if class_id == 'road':
                alpha = 0.4  # 透明度参数，范围从0（完全透明）到1（完全不透明）
            else:
                alpha = 0.5
            for i in range(3):  # 对于每个颜色通道（B, G, R）
                original_image[..., i] = np.where(decoded_mask == 1,
                                                  original_image[..., i] * (1 - alpha) + color[i] * alpha,
                                                  original_image[..., i])
            if class_id != 'background':
                cv2.putText(original_image, class_id, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 1,
                        cv2.LINE_AA)
        file_name_without_extension, file_extension = os.path.splitext(filename)
        imgurl = os.path.join(directory_path.replace('annotations', 'mask'), file_name_without_extension + '.jpg')
        cv2.imencode('.png', original_image)[1].tofile(imgurl)
        print('Saved mask image to {}'.format(imgurl))
        # cv2.imencode('Decoded Mask', merged_image * 255)  # 将二值图像乘以255以便于显示

    # counts_list.append(counts)
    pass
    # 提取 counts
if __name__ == '__main__':
    directory_path = r'Q:\XSS\sam_dataset\dataset4.3\dataset4.3.1\annotations'
    frPyObjects(directory_path)