from ultralytics import YOLO
# Load a model
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

# Load a model
    model = YOLO(r"E:\FrankIce\Pycharm_coder\ultralytics_guard_bar\runs\classify\train13\weights\best.pt")  # load the trained model

    # Validate the model
    metrics = model.val(batch=1, imgsz=300)  # no arguments needed, uses the dataset and settings from training
    metrics.top1  # top1 accuracy
    metrics.top5  # top5 accuracy