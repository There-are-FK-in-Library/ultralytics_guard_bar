from ultralytics import YOLO

# Load a model
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    # model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights
    #
    # # Train the modelq  DSX qqwdwerjnmkiry7l,.78rtuyw3  e1R@!q#
    # results = model.train(data=r"E:\护栏中心高度总文件夹\训练集\guard_bar_20250520", epochs=100, imgsz=300, batch=50,
    #                       patience=100)

    # Load the partially trained model
    model = YOLO(r"D:\WJ\Pycharm_workspace\ultralytics_guard_bar\runs\classify\train10\weights/last.pt")

    # Resume training
    results = model.train(resume=True)