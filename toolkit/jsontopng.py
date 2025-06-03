import argparse
import base64
import json
import os
import os.path as osp

import PIL.Image
import yaml

import logging
logger = logging.getLogger(__name__)
from labelme import utils

path = r"Q:\XSS\sam_dataset\dataset4.3\dataset4.3.1\annotations"
dirs = os.listdir(path)


def label(json_file, out_dir, label_name_to_value):
    data = json.load(open(json_file))[0]
    if data['imageData']:
        imageData = data['imageData']
    else:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = utils.img_b64_to_arr(imageData)

    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name
    lbl_viz = utils.draw_label(lbl, img, label_names)

    PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
    utils.lblsave(osp.join(out_dir, 'label.png'), lbl)
    PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))

    with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
        for lbl_name in label_names:
            f.write(lbl_name + '\n')

    logger.warning('info.yaml is being replaced by label_names.txt')
    info = dict(label_names=label_names)
    with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
        yaml.safe_dump(info, f, default_flow_style=False)

    logger.info('Saved to: {}'.format(out_dir))


def main():
    logger.warning('This script is aimed to demonstrate how to convert the'
                   'JSON file to a single image dataset, and not to handle'
                   'multiple JSON files to generate a real-use dataset.')

    # parser = argparse.ArgumentParser()
    # parser.add_argument(r'Q:\XSS\sam_dataset\dataset4.3\dataset4.3.1\annotations')
    # parser.add_argument( r'Q:\XSS\sam_dataset\dataset4.3\dataset4.3.1\out', default=None)
    # args = parser.parse_args()
    label_name_to_value = {'_background_': 0}
    json_file_root = r'Q:\XSS\sam_dataset\dataset4.3\dataset4.3.1\annotations'
    out_dir =  r'Q:\XSS\sam_dataset\dataset4.3\dataset4.3.1\out'
    for json_file in dirs:
        # 使用 os.path.join 构造完整的文件路径
        json_file_path = os.path.join(json_file_root, json_file)
        # 输出文件的目标路径
        # output_file_path = os.path.join(out_dir, json_file)  # 如果需要将文件名保留到输出目录

        # 调用 label 函数
        label(json_file_path, out_dir, label_name_to_value)


if __name__ == '__main__':
    main()

