from ultralytics import YOLO

# 모델 학습 경로 및 설정
model_path = 'C:/Users/asd/Desktop/colony.v1i.yolov8/yolov8n.pt'
data_path = 'C:/Users/asd/Desktop/colony.v1i.yolov8/data.yaml'
save_dir = 'C:/Users/asd/Desktop/colony.v1i.yolov8/models/trained_model.pt'

# YOLO 모델 로드 (pretrained 모델을 사용할 수 있습니다)
model = YOLO(model_path)

# 모델 학습
model.train(data=data_path, epochs=10, batch=32, imgsz=500, project='C:/Users/asd/Desktop/colony.v1i.yolov8/runs', name='trained_model', save=True)

# 학습이 완료되면 학습된 모델이 `project` 폴더에 저장됩니다.
