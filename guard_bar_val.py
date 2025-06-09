from ultralytics import YOLO

# Load a model
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

# Load a model
    model = YOLO("runs/classify/train7/weights/best.pt")  # load the trained model

    # Validate the model
    metrics = model.val(batch=2)  # no arguments needed, uses the dataset and settings from training
    metrics.top1  # top1 accuracy
    metrics.top5  # top5 accuracy