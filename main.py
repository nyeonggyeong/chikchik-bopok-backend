from io import BytesIO
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="ChikChik-Bopok Object Detection API")

# 가장 가벼운 YOLO11 가중치 사용 (없으면 ultralytics가 자동 다운로드)
MODEL_WEIGHTS = "yolo11n.pt"
model = YOLO(MODEL_WEIGHTS)


@app.post("/predict/objects")
async def predict_objects(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    try:
        raw_bytes = await file.read()
        pil_image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="유효한 이미지 파일이 아닙니다.") from exc

    image_width, image_height = pil_image.size
    image_area = float(image_width * image_height)

    # PIL 이미지를 바로 YOLO에 전달하여 추론
    results = model(pil_image, verbose=False)
    if not results:
        return {
            "filename": file.filename,
            "image_size": {"width": image_width, "height": image_height},
            "objects": [],
            "total_objects": 0,
        }

    result = results[0]
    names = result.names
    detected_objects: List[Dict[str, Any]] = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0].item())
        class_id = int(box.cls[0].item())
        class_name = names[class_id] if class_id in names else str(class_id)

        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        bbox_area = width * height
        area_ratio_percent = (bbox_area / image_area) * 100 if image_area > 0 else 0.0

        detected_objects.append(
            {
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": {
                    "x": round(float(x1), 2),
                    "y": round(float(y1), 2),
                    "w": round(float(width), 2),
                    "h": round(float(height), 2),
                },
                "area_ratio_percent": round(area_ratio_percent, 2),
                "is_over_30_percent": area_ratio_percent > 30.0,
            }
        )

    return {
        "filename": file.filename,
        "image_size": {"width": image_width, "height": image_height},
        "objects": detected_objects,
        "total_objects": len(detected_objects),
    }
