import os
import numpy as np
import cv2
from PIL import Image
import time
import multiprocessing
import json
import sys

with open(r'D:\WJ\Pycharm_workspace\ultralytics_guard_bar\configs\config.json', 'r') as f:
    tmp = f.read()
    configDic = json.loads(tmp)

def change_bgr_hsv_v(img, v_value=105):
    H, W = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mean = np.mean(v)

    base_mean = v_value
    factor = base_mean / mean

    blank = np.zeros((H, W, 3), dtype=np.uint8)
    new_img = cv2.addWeighted(img, factor, blank, 0, 0)
    return new_img

class ByteTrackArgs():
    def __init__(self) -> None:
        self.track_thresh = 0.25  # 高低分匹配的分界线
        self.track_buffer = 30  # 跟踪的对象数量
        self.match_thresh = 0.7
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = False
        self.tsize = None
        self.name = 'yolo'


if __name__ == '__main__':

    t = 0