#!/usr/bin/python3
# coding=utf-8
import math
import multiprocessing
from multiprocessing import Manager, Process, Lock, set_start_method
from multiprocessing import Queue as Multi_Queue
# from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_convnext, ft_net_efficient, ft_net_NAS, PCB

import time
from flask import Flask, request, jsonify
import time
from ultralytics import YOLO
import json
import cv2
import base64
from gevent import pywsgi
from user_utils import *
from user_class import *
# from own_predict import SmallSignsClassify
from multiprocessing import freeze_support


import numpy as np

def base64_to_cv2(image_base64):
    """base64 image to cv2"""
    image_bytes = base64.b64decode(image_base64)
    np_array = np.fromstring(image_bytes, np.uint8)
    image_cv2 = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image_cv2

def mat_to_base64(value):
    return base64.b64encode(cv2.imencode('.png', value)[1]).decode()


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


def change_rgb_hsv_v(img, v_value=105):
    H, W = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mean = np.mean(v)

    base_mean = v_value
    factor = base_mean / mean

    blank = np.zeros((H, W, 3), dtype=np.uint8)
    new_img = cv2.addWeighted(img, factor, blank, 0, 0)
    return new_img


def put_time_str_bgr(img):
    message = time.strftime("%Y/%m/%d %H:%M:%S")
    cv2.putText(img, message, (540, 520), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 153, 255), 2)
    return img


def put_time_str_rgb(img):
    message = time.strftime("%Y/%m/%d %H:%M:%S")
    cv2.putText(img, message, (540, 520), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 153, 0), 2)
    return img


with open(r'D:\WJ\Pycharm_workspace\ultralytics_guard_bar\configs\config.json', 'r') as f:
    tmp = f.read()
    configDic = json.loads(tmp)


# app.debug = True
class FlaskServerLaneClass():
    def __init__(self):
        pass

    def run(self):

        # 传输载体： 图像文件名
        #(multiprocessing.Process(target=self.flask_detection_server_run_file)).start()
        # 传输载体： HImage-3通道
        #(multiprocessing.Process(target=self.flask_detection_server_run)).start()

        for i in range(5100, 5300, 100):
            print(fr'----index {i}')
            (multiprocessing.Process(target=self.flask_detection_server_run_port, args=(i,))).start()
            time.sleep(10)




    # def flask_detection_server_run_file(self):
    #     status = False
    #     app = Flask('flask_bino_server')
    #     mlog = logger('xex_flask_bino_server.log')
    #     detector = SmallSignsClassify()
    #
    #     mlog.info("detection init Net success")
    #     status = True
    #
    #     @app.route('/http/url_get_bino_obj_detect_from_py/', methods=['POST'])
    #     def post_http_img_data():
    #         remote_ip = request.remote_addr
    #         if not request.data:  # 检测是否有数据
    #             return ('fail')
    #         t0 = time.time()
    #         body = request.data.decode('utf-8')
    #         # 获取到POST过来的数据，因为我这里传过来的数据需要转换一下编码。根据晶具体情况而定
    #         bodyDic = json.loads(body)
    #         filename = bodyDic['imgL']
    #         images = []
    #         img = Image.open(filename).convert('RGB')
    #         images.append(img)
    #
    #         # 预测 结果
    #         results, time_out = detector.Runs(images)
    #         value_s, index_s = results.max(1)
    #
    #         out_dic = {'predict': []}
    #         out_dic['predict'].append(index_s[0].item())
    #
    #         t1 = time.time()
    #         mlog.info("finish bino task remote({0}),use time({1} ms)".format(remote_ip, int((t1-t0)*1000)))
    #         return jsonify(out_dic)
    #
    #     @app.route('/http/url_check_bino_status_from_py/', methods=['POST'])
    #     def check_status():
    #         remote_ip = request.remote_addr
    #         mlog.info("remote({0}) check bino status: status".format(remote_ip))
    #         return jsonify({"status": status})
    #
    #     pywsgi.WSGIServer((configDic["py_server_ip"], 5000), app, log=None).serve_forever()


    # def flask_detection_server_run(self):
    #     status = False
    #     app = Flask('flask_bino_server')
    #     mlog = logger('xex_flask_bino_server.log')
    #     detector = SmallSignsClassify()
    #
    #     mlog.info("detection init Net success")
    #     status = True
    #
    #     @app.route('/http/url_get_bino_obj_detect_from_py/', methods=['POST'])
    #     def post_http_img_data():
    #         remote_ip = request.remote_addr
    #         if not request.data:  # 检测是否有数据
    #             return ('fail')
    #         t0 = time.time()
    #         body = request.data.decode('utf-8')
    #         # 获取到POST过来的数据，因为我这里传过来的数据需要转换一下编码。根据晶具体情况而定
    #         bodyDic = json.loads(body)
    #         img = base64_to_cv2(bodyDic['imgL'])
    #         image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #         images = []
    #         images.append(image)
    #
    #         # 预测 结果
    #         results, time_out = detector.Runs(images)
    #         value_s, index_s = results.max(1)
    #
    #         out_dic = {'predict': []}
    #         out_dic['predict'].append(index_s[0].item())
    #
    #         t1 = time.time()
    #         mlog.info("finish bino task remote({0}),use time({1} ms)".format(remote_ip, int((t1-t0)*1000)))
    #         return jsonify(out_dic)
    #
    #     @app.route('/http/url_check_bino_status_from_py/', methods=['POST'])
    #     def check_status():
    #         remote_ip = request.remote_addr
    #         mlog.info("remote({0}) check bino status: status".format(remote_ip))
    #         return jsonify({"status":status})
    #
    #     pywsgi.WSGIServer((configDic["py_server_ip"], 5100), app, log=None).serve_forever()


    def flask_detection_server_run_port(self, port):
        status = False
        app = Flask('flask_bino_server')
        mlog = logger('xex_flask_bino_server.log')
        # detector = SmallSignsClassify()

        # mlog.info("detection init Net success")
        status = True
        model = YOLO(r"D:\WJ\Pycharm_workspace\ultralytics_guard_bar\runs\classify\train13\weights\best.pt")

        @app.route('/http/url_get_bino_obj_detect_from_py/', methods=['POST'])
        def post_http_img_data():
            remote_ip = request.remote_addr
            if not request.data:  # 检测是否有数据
                return ('fail')
            t0 = time.time()
            body = request.data.decode('utf-8')
            # 获取到POST过来的数据，因为我这里传过来的数据需要转换一下编码。根据晶具体情况而定
            bodyDic = json.loads(body)
            img = base64_to_cv2(bodyDic['imgL'])
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            images = []
            images.append(image)

            # 预测 结果
            # results, time_out = detector.Runs(images)
            # value_s, index_s = results.max(1)
            results = model(images, save=False, batch=100)

            out_dic = {'predict': []}
            # out_dic['predict'].append(index_s[0].item())
            out_dic['predict'].append(results[0].probs.top1)

            t1 = time.time()
            mlog.info("finish bino task remote({0}),use time({1} ms)".format(remote_ip, int((t1-t0)*1000)))
            return jsonify(out_dic)

        @app.route('/http/url_check_bino_status_from_py/', methods=['POST'])
        def check_status():
            remote_ip = request.remote_addr
            mlog.info("remote({0}) check bino status: status".format(remote_ip))
            return jsonify({"status":status})

        pywsgi.WSGIServer((configDic["py_server_ip"], port), app, log=None).serve_forever()


if __name__ == '__main__':

    multiprocessing.freeze_support()
    freeze_support()
    app_lane_server = FlaskServerLaneClass()
    app_lane_server.run()


    while True:
        time.sleep(1000)
    # app.run(host='127.0.0.1', port=5000,debug=True)
    # 这里指定了地址和端口号。也可以不指定地址填0.0.0.0那么就会使用本机地址ip

