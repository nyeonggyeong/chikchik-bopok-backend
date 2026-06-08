import os
import shutil
from ultralytics import YOLO

def main():
    # 데이터셋이 이미 다운로드되어 있으므로 바로 학습 시작
    data_yaml_path = os.path.join("stairs-3", "data.yaml")

    if not os.path.exists(data_yaml_path):
        print(f"[오류] '{data_yaml_path}' 파일을 찾을 수 없습니다.")
        return

    print(f"[학습 시작] 데이터셋: {data_yaml_path}")
    print("CPU 환경에서 50 에포크 학습을 진행합니다. 약 30분~2시간 소요될 수 있습니다.\n")

    model = YOLO('yolo11n.pt')

    try:
        model.train(
            data=data_yaml_path,
            epochs=50,
            imgsz=640,
            batch=8,
            name='stairs_custom_model',
            device='cpu'
        )
        print("\n[완료] 학습이 성공적으로 완료되었습니다!")

        best_pt_path = os.path.join("runs", "detect", "stairs_custom_model", "weights", "best.pt")
        if os.path.exists(best_pt_path):
            os.makedirs("weights", exist_ok=True)
            target_path = os.path.join("weights", "best.pt")
            shutil.copy(best_pt_path, target_path)
            print(f"[성공] 가중치 파일이 '{target_path}'에 저장되었습니다!")
            print("이제 detection.py의 MODEL_WEIGHTS = 'weights/best.pt' 로 변경하면 됩니다.")
        else:
            print(f"[경고] '{best_pt_path}' 파일을 찾을 수 없습니다.")

    except Exception as e:
        print(f"\n[오류] 학습 중 문제 발생: {e}")

if __name__ == '__main__':
    main()
