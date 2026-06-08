import os
from ultralytics import YOLO

def main():
    print("🚀 YOLOv11 커스텀 학습(Fine-tuning)을 시작합니다!")
    print("데이터셋 폴더 내에 'data.yaml' 파일이 정상적으로 존재하는지 확인해 주세요.\n")

    # 1. 사용할 데이터셋 설정 파일 경로 (Roboflow 등에서 다운로드한 폴더 내 data.yaml)
    # 현재 디렉토리에 data.yaml이 있다고 가정합니다. 경로가 다르다면 수정해 주세요.
    data_yaml_path = 'data.yaml'

    if not os.path.exists(data_yaml_path):
        print(f"❌ 오류: '{data_yaml_path}' 파일을 찾을 수 없습니다.")
        print("Roboflow에서 다운받은 데이터셋 압축을 해제하고, 이 스크립트와 동일한 폴더에 위치시켜 주세요.")
        return

    try:
        # 2. 사전 학습된 기본 모델 로드
        print("📦 기본 모델(yolo11n.pt)을 불러옵니다...")
        model = YOLO('yolo11n.pt')

        # 3. 모델 학습 시작
        # - data: 데이터셋 설정 파일
        # - epochs: 학습 반복 횟수 (보통 50~100 사이가 적당합니다)
        # - imgsz: 입력 이미지 크기
        # - batch: 한 번에 처리할 이미지 수 (메모리 부족 시 줄여주세요)
        print("🏃 학습을 시작합니다. 이 작업은 GPU/CPU 성능에 따라 수십 분이 소요될 수 있습니다.")
        results = model.train(
            data=data_yaml_path,
            epochs=50,
            imgsz=640,
            batch=16,
            name='stairs_custom_model', # 학습 결과가 저장될 폴더 이름
            device='auto' # 사용 가능한 GPU가 있으면 자동 할당
        )

        print("\n✅ 학습이 완료되었습니다!")
        print("생성된 가중치 파일(best.pt)은 'runs/detect/stairs_custom_model/weights/' 폴더에 저장되어 있습니다.")
        print("해당 파일을 weights 폴더로 복사하고, detection.py의 MODEL_WEIGHTS를 변경하여 사용하세요.")

    except Exception as e:
        print(f"\n❌ 학습 중 오류가 발생했습니다: {e}")

if __name__ == '__main__':
    main()
