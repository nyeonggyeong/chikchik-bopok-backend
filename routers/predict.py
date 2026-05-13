from typing import Any, Dict, List, Literal, Optional

import time
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from services.depth import LITEMONO_MODEL_NAME, depth_estimator
from services.detection import _extract_objects, _read_image, model
from services.spatial_analysis import OVERLAP_RATIO_THRESHOLD_DEFAULT, analyze_spatial_results, detections_from_yolo
from services.guide_service import guide_service

router = APIRouter(prefix="/predict", tags=["predict"])


class SpatialAnalysisObject(BaseModel):
    label: str
    label_ko: str = ""
    confidence: float
    position: str
    position_ko: str = ""
    distance: str
    distance_text: str = ""
    is_empty: Optional[bool] = None
    description: str
    x1: float
    y1: float
    x2: float
    y2: float
    estimated_distance_m: float
    raw_depth_value: float = 0.0
    reference_depth: float = 1.0
    distance_confidence: str = "medium"
    risk_level: int
    motion_state: str = "stable"


class PredictObjectsSpatialResponse(BaseModel):
    risk_level: str
    main_hazard: str
    safe_direction: str
    guide_message: str = ""
    guide_source: str = "none"
    process_time: str = "0s"
    display_objects: List[SpatialAnalysisObject]
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
    reference_depth: float = Form(1.0),
) -> PredictObjectsSpatialResponse:
    start_time = time.time()
    pil_image = await _read_image(file)
    read_time = time.time() - start_time
    
    image_width, image_height = pil_image.size
    
    if not depth_estimator.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Lite-Mono가 준비되지 않았습니다. "
                f"사유: {depth_estimator.error_message}"
            ),
        )

    # YOLO Detection
    yolo_start = time.time()
    results = model(pil_image, verbose=False, conf=0.15)
    yolo_time = time.time() - yolo_start

    if not results:
        return PredictObjectsSpatialResponse(
            risk_level="safe",
            main_hazard="감지된 위험 요소 없음",
            safe_direction="forward",
            display_objects=[],
            objects=[]
        )

    # Lite-Mono Depth
    depth_start = time.time()
    depth_map = depth_estimator.predict_depth_map(pil_image)
    depth_time = time.time() - depth_start

    h, w = depth_map.shape

    # Spatial Analysis
    spatial_start = time.time()
    detection_list = detections_from_yolo(results[0])
    analyzed_data = analyze_spatial_results(
        detection_list,
        depth_map,
        w,
        h,
        overlap_threshold=overlap_threshold,
        danger_threshold=danger_threshold,
        reference_depth=reference_depth,
    )
    spatial_time = time.time() - spatial_start

    objects = [
        SpatialAnalysisObject(
            label=o["label"],
            label_ko=o.get("label_ko", o["label"]),
            confidence=o["confidence"],
            position=o["position"],
            position_ko=o.get("position_ko", "전방"),
            distance=o["distance"],
            distance_text=o.get("distance_text", o["distance"]),
            is_empty=o["is_empty"],
            description=o["description"],
            x1=o["x1"],
            y1=o["y1"],
            x2=o["x2"],
            y2=o["y2"],
            estimated_distance_m=o["estimated_distance_m"],
            raw_depth_value=o.get("raw_depth_value", 0.0),
            reference_depth=reference_depth,
            distance_confidence=o.get("distance_confidence", "medium"),
            risk_level=o.get("risk_level", 0),
            motion_state=o.get("motion_state", "stable")
        )
        for o in analyzed_data
    ]

    # Phase 5.6: 안내 대상 객체 선정 (최대 2개)
    # 위험도 높은 순으로 1차 필터링
    risky_objects = [o for o in objects if o.risk_level > 0]
    
    # 만약 위험 객체가 1개 이하라면, 회피 방향에 영향을 주는 다른 객체도 후보로 포함
    if len(risky_objects) < 2:
        safe_objects = [o for o in objects if o.risk_level == 0]
        # 회피 방향(safe_dir 계산 전이지만 위치로 추정) 근처 객체나 전방 객체 추가
        risky_objects.extend(safe_objects[:2 - len(risky_objects)])

    display_objects = risky_objects[:2]
    
    main_hazard = "감지된 위험 요소 없음"
    risk_level_str = "safe"
    if display_objects:
        top_obj = display_objects[0]
        risk_level_str = "danger" if top_obj.risk_level == 2 else ("warning" if top_obj.risk_level == 1 else "safe")
        main_hazard = f"{top_obj.position_ko} {top_obj.label_ko}"

    from services.spatial_analysis import calculate_safe_direction
    safe_dir = calculate_safe_direction(analyzed_data)

    # Gemini 가이드 생성 (Phase 5.6: 모든 상세 필드 전달)
    guide_result = await guide_service.generate_guide(
        risk_level=risk_level_str,
        main_hazard=main_hazard,
        safe_direction=safe_dir,
        display_objects=[o.dict() for o in display_objects]
    )

    total_time = time.time() - start_time
    print(f"⏱️ [Total: {total_time:.3f}s] Read: {read_time:.3f}s, YOLO: {yolo_time:.3f}s, Depth: {depth_time:.3f}s, Spatial: {spatial_time:.3f}s, Guide: {guide_result.get('process_time', '0s')}")

    return PredictObjectsSpatialResponse(
        risk_level=risk_level_str,
        main_hazard=main_hazard,
        safe_direction=safe_dir,
        guide_message=guide_result.get("guide_message", ""),
        guide_source=guide_result.get("guide_source", "none"),
        process_time=f"{total_time:.3f}s",
        display_objects=display_objects,
        objects=objects
    )
