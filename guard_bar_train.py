from ultralytics import YOLO

# Load a model
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=r"E:/护栏中心高度总文件夹/guard_bar_data/", epochs=100, imgsz=320, batch=24,
                          patience=50)