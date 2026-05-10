from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from services.depth import LITEMONO_MODEL_NAME, depth_estimator
from services.detection import _extract_objects, _read_image, model
from services.spatial_analysis import OVERLAP_RATIO_THRESHOLD_DEFAULT, analyze_spatial_results, detections_from_yolo

router = APIRouter(prefix="/predict", tags=["predict"])


class SpatialAnalysisObject(BaseModel):
    label: str
    confidence: float
    position: Literal["왼쪽", "중앙", "오른쪽"]
    distance: str
    is_empty: Optional[bool] = None
    description: str
    x1: float
    y1: float
    x2: float
    y2: float
    distance_estimate_m: float


class PredictObjectsSpatialResponse(BaseModel):
    objects: List[SpatialAnalysisObject]


@router.post("/objects")
async def predict_objects(file: UploadFile = File(...)) -> Dict[str, Any]:
    pil_image = await _read_image(file)

    image_width, image_height = pil_image.size
    image_area = float(image_width * image_height)

    results = model(pil_image, verbose=False)
    if not results:
        return {
            "filename": file.filename,
            "image_size": {"width": image_width, "height": image_height},
            "objects": [],
            "total_objects": 0,
        }

    result = results[0]
    detected_objects = _extract_objects(result, image_width=float(image_width), image_height=float(image_height))

    return {
        "filename": file.filename,
        "image_size": {"width": image_width, "height": image_height},
        "objects": detected_objects,
        "total_objects": len(detected_objects),
    }


@router.post("/objects-distance", response_model=PredictObjectsSpatialResponse)
async def predict_objects_distance(
    file: UploadFile = File(...),
    danger_threshold: float = Form(1.5),
    overlap_threshold: float = Form(OVERLAP_RATIO_THRESHOLD_DEFAULT),
) -> PredictObjectsSpatialResponse:
    pil_image = await _read_image(file)
    image_width, image_height = pil_image.size
    image_area = float(image_width * image_height)
    if not depth_estimator.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Lite-Mono가 준비되지 않았습니다. "
                f"사유: {depth_estimator.error_message}"
            ),
        )

    results = model(pil_image, verbose=False)
    if not results:
        return PredictObjectsSpatialResponse(objects=[])

    depth_map = depth_estimator.predict_depth_map(pil_image)
    # depth는 원본 이미지 크기로 보간되어 (H, W) = 이미지 세로×가로
    h, w = depth_map.shape

    detection_list = detections_from_yolo(results[0])
    analyzed_data = analyze_spatial_results(
        detection_list,
        depth_map,
        w,
        h,
        overlap_threshold=overlap_threshold,
        danger_threshold=danger_threshold,
    )

    objects = [
        SpatialAnalysisObject(
            label=o["label"],
            confidence=o["confidence"],
            position=o["position"],
            distance=o["distance"],
            is_empty=o["is_empty"],
            description=o["description"],
            x1=o["x1"],
            y1=o["y1"],
            x2=o["x2"],
            y2=o["y2"],
            distance_estimate_m=o["distance_estimate_m"],
        )
        for o in analyzed_data
    ]

    return PredictObjectsSpatialResponse(objects=objects)
