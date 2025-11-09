from ultralytics import YOLO

model = YOLO('yolov12s.pt')
model.val(data='coco.yaml', save_json=True)