from io import BytesIO
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from fastapi import HTTPException, UploadFile
from ultralytics import YOLO

# 가장 가벼운 YOLO11 가중치 사용 (없으면 ultralytics가 자동 다운로드)
MODEL_WEIGHTS = "yolo11n.pt"
model = YOLO(MODEL_WEIGHTS)


def _extract_objects(result: Any, image_width: float, image_height: float, depth_map: np.ndarray | None = None, danger_threshold: float = 1.5) -> List[Dict[str, Any]]:
    names = result.names
    detected_objects: List[Dict[str, Any]] = []
    image_area = image_width * image_height

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0].item())
        class_id = int(box.cls[0].item())
        class_name = names[class_id] if class_id in names else str(class_id)

        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        bbox_area = width * height
        area_ratio_percent = (bbox_area / image_area) * 100 if image_area > 0 else 0.0

        x_norm = x1 / image_width if image_width > 0 else 0.0
        y_norm = y1 / image_height if image_height > 0 else 0.0
        w_norm = width / image_width if image_width > 0 else 0.0
        h_norm = height / image_height if image_height > 0 else 0.0

        obj: Dict[str, Any] = {
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "bbox": {
                "x": round(float(x_norm), 4),
                "y": round(float(y_norm), 4),
                "w": round(float(w_norm), 4),
                "h": round(float(h_norm), 4),
            },
            "area_ratio_percent": round(area_ratio_percent, 2),
            "is_over_30_percent": area_ratio_percent > 30.0,
        }

        if depth_map is not None:
            h, w = depth_map.shape
            sx1 = max(0, min(int(x1), w - 1))
            sy1 = max(0, min(int(y1), h - 1))
            sx2 = max(0, min(int(x2), w))
            sy2 = max(0, min(int(y2), h))

            if sx2 > sx1 and sy2 > sy1:
                roi = depth_map[sy1:sy2, sx1:sx2]
                distance_m = float(np.median(roi))
                if distance_m < 1.5:
                    distance_level = "near"
                elif distance_m < 3.0:
                    distance_level = "mid"
                else:
                    distance_level = "far"
                obj["distance_estimate_m"] = round(distance_m, 2)
                obj["distance_level"] = distance_level
                obj["is_dangerous"] = bool(obj["area_ratio_percent"] > 20.0 and distance_m <= danger_threshold)
            else:
                obj["distance_estimate_m"] = None
                obj["distance_level"] = "unknown"
                obj["is_dangerous"] = bool(obj["is_over_30_percent"])

        detected_objects.append(obj)

    return detected_objects


async def _read_image(file: UploadFile) -> Image.Image:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    try:
        raw_bytes = await file.read()
        return Image.open(BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="유효한 이미지 파일이 아닙니다.") from exc
