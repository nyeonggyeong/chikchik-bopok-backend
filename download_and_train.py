import os
import subprocess
import sys
import shutil

def main():
    # .env 파일에서 환경변수 로드
    from dotenv import load_dotenv
    load_dotenv()

    api_key   = os.getenv("ROBOFLOW_API_KEY")
    workspace = os.getenv("ROBOFLOW_WORKSPACE")
    project   = os.getenv("ROBOFLOW_PROJECT")
    version   = int(os.getenv("ROBOFLOW_VERSION", "3"))

    if not api_key:
        print("[오류] .env 파일에 ROBOFLOW_API_KEY가 없습니다.")
        return

    print("[Step 1] 필요한 패키지 설치 중...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow", "ultralytics", "python-dotenv"])

    print("\n[Step 2] Roboflow에서 계단 데이터셋 다운로드 중...")
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov11")

    print(f"\n데이터셋 다운로드 완료! 위치: {dataset.location}")
    data_yaml_path = os.path.join(dataset.location, "data.yaml")

    print("\n[Step 3] YOLOv11 커스텀 학습 시작 (50 에포크, 약 30분~1시간 소요)...")
    from ultralytics import YOLO
    model = YOLO('yolo11n.pt')

    try:
        model.train(
            data=data_yaml_path,
            epochs=50,
            imgsz=640,
            batch=16,
            name='stairs_custom_model',
            device='auto'
        )
        print("\n[완료] 학습이 성공적으로 완료되었습니다!")

        # best.pt를 weights/ 폴더로 자동 복사
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
