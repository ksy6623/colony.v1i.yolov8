import os
from ultralytics import YOLO

# 수정된 경로
trained_model_path = r'C:\Users\asd\Desktop\colony.v1i.yolov8\runs\trained_model3\weights\best.pt'
source_path = r'C:\Users\asd\Desktop\colony.v1i.yolov8\Original_img'  # 예측할 이미지가 있는 폴더
save_path = r'C:\Users\asd\Desktop\colony.v1i.yolov8\Predicted_images'  # 예측 결과를 저장할 폴더

# 경로 존재 여부 확인 및 디렉토리 생성
if not os.path.exists(source_path):
    raise FileNotFoundError(f"The source path '{source_path}' does not exist.")
if not os.path.exists(save_path):
    os.makedirs(save_path)  # 저장할 폴더가 없으면 생성

# YOLO 모델 로드
model = YOLO(trained_model_path)

# 모델 예측 수행
results = model.predict(source=source_path, save=True)

# 결과 저장 경로 확인 및 파일 이동
for result in results:
    if result.path:
        file_name = os.path.basename(result.path)  # 원본 파일 이름 추출
        save_file_path = os.path.join(save_path, file_name)  # 저장할 경로

        # 예측된 이미지를 지정된 폴더로 저장
        result.save(save_path)  # 결과를 지정된 폴더로 저장
        
        # 저장된 이미지의 경로를 출력
        print(f"Saved result to {os.path.join(save_path, file_name)}")

    # 이미지 출력 (옵션, 삭제 가능)
    result.show()
