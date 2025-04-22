
import logging
import time
import os

def logger(file_log):
    mdir = './logs'
    mdate = time.strftime("%Y%m%d")

    log_dir = '{0}/{1}'.format(mdir, mdate)
    os.makedirs(log_dir, exist_ok=True)
    log_path = '{0}/{1}'.format(log_dir, file_log)
    '''
    打日志
    :param file_log: 日志文件名，类型string；
    NOTSET（0）、DEBUG（10）、INFO（20）、WARNING（30）、ERROR（40）、CRITICAL（50）
    '''
    # 创建一个loggger，并设置日志级别
    logger = logging.getLogger()
    # 低于该级别的将忽略
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于写入日志文件，并设置日志级别，mode:a是追加写模式，w是覆盖写模式
    fh = logging.FileHandler(filename=log_path, encoding='utf-8', mode='a')
    # 低于该级别的将忽略
    fh.setLevel(logging.INFO)

    # 创建一个handler，用于将日志输出到控制台，并设置日志级别
    ch = logging.StreamHandler()
    # 低于该级别的将忽略
    ch.setLevel(logging.INFO)

    # 定义handler的输出格式
    # formatter = logging.Formatter(
    #     '%(asctime)s-%(name)s-%(filename)s-[line:%(lineno)d]''-%(levelname)s-[日志信息]: %(message)s')
    formatter = logging.Formatter(
        '%(asctime)s-[line:%(lineno)d]''-%(levelname)s-[日志信息]: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

