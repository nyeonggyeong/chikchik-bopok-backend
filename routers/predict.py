import json
import time
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel

from services.depth import LITEMONO_MODEL_NAME, depth_estimator
from services.detection import _extract_objects, _read_image, model
from services.spatial_analysis import OVERLAP_RATIO_THRESHOLD_DEFAULT, analyze_spatial_results, detections_from_yolo, calculate_safe_direction
from services.guide_service import guide_service, build_hazard_summary

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

    # Phase 5.6: 안내 대상 객체 선정 (최대 3개)
    # 위험도 높은 순으로 1차 필터링
    risky_objects = [o for o in objects if o.risk_level > 0]
    
    # 만약 위험 객체가 2개 이하라면, 회피 방향에 영향을 주는 다른 객체도 후보로 포함
    if len(risky_objects) < 3:
        safe_objects = [o for o in objects if o.risk_level == 0]
        # 회피 방향(safe_dir 계산 전이지만 위치로 추정) 근처 객체나 전방 객체 추가
        risky_objects.extend(safe_objects[:3 - len(risky_objects)])

    display_objects = risky_objects[:3]
    
    main_hazard = "감지된 위험 요소 없음"
    risk_level_str = "safe"
    if display_objects:
        top_obj = display_objects[0]
        risk_level_str = "danger" if top_obj.risk_level == 2 else ("warning" if top_obj.risk_level == 1 else "safe")
        main_hazard = build_hazard_summary([o.dict() for o in display_objects])

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


def _build_spatial_objects(
    analyzed_data: List[Dict[str, Any]],
    reference_depth: float,
) -> List[SpatialAnalysisObject]:
    """analyze_spatial_results 결과를 SpatialAnalysisObject 리스트로 변환하는 공통 헬퍼."""
    return [
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
            motion_state=o.get("motion_state", "stable"),
        )
        for o in analyzed_data
    ]


def _select_display_objects(
    objects: List[SpatialAnalysisObject],
) -> List[SpatialAnalysisObject]:
    """위험도 기준으로 최대 3개 안내 객체 선정."""
    risky = [o for o in objects if o.risk_level > 0]
    if len(risky) < 3:
        safe = [o for o in objects if o.risk_level == 0]
        risky.extend(safe[: 3 - len(risky)])
    return risky[:3]


@router.websocket("/ws")
async def predict_websocket(
    websocket: WebSocket,
    danger_threshold: float = 1.5,
    overlap_threshold: float = OVERLAP_RATIO_THRESHOLD_DEFAULT,
    reference_depth: float = 1.0,
) -> None:
    """
    WebSocket 실시간 스트리밍 엔드포인트.

    클라이언트 → 서버: 이미지 바이너리(JPEG/PNG) 전송
    서버 → 클라이언트: PredictObjectsSpatialResponse JSON 문자열 push

    연결 URL 예시:
        ws://<host>/predict/ws?danger_threshold=1.5&overlap_threshold=0.3&reference_depth=1.0
    """
    await websocket.accept()
    print("🔌 [WebSocket] 클라이언트 연결됨")

    try:
        while True:
            # 1. 클라이언트에서 이미지 바이너리 수신
            raw_bytes = await websocket.receive_bytes()

            frame_start = time.time()

            # 2. 이미지 디코딩
            try:
                pil_image = Image.open(BytesIO(raw_bytes)).convert("RGB")
            except Exception as exc:
                await websocket.send_text(
                    json.dumps({"error": f"이미지 디코딩 실패: {exc}"}, ensure_ascii=False)
                )
                continue

            # 3. Depth 모델 준비 확인
            if not depth_estimator.is_ready:
                await websocket.send_text(
                    json.dumps(
                        {"error": f"Lite-Mono 미준비: {depth_estimator.error_message}"},
                        ensure_ascii=False,
                    )
                )
                continue

            image_width, image_height = pil_image.size

            # 4. YOLO 탐지
            results = model(pil_image, verbose=False, conf=0.15)
            if not results:
                empty_resp = PredictObjectsSpatialResponse(
                    risk_level="safe",
                    main_hazard="감지된 위험 요소 없음",
                    safe_direction="forward",
                    process_time=f"{time.time() - frame_start:.3f}s",
                    display_objects=[],
                    objects=[],
                )
                await websocket.send_text(
                    json.dumps(empty_resp.dict(), ensure_ascii=False)
                )
                continue

            # 5. Lite-Mono 깊이 추정
            depth_map = depth_estimator.predict_depth_map(pil_image)
            h, w = depth_map.shape

            # 6. 공간 분석
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

            objects = _build_spatial_objects(analyzed_data, reference_depth)
            display_objects = _select_display_objects(objects)

            # 7. 위험도 및 주요 위험 요약
            main_hazard = "감지된 위험 요소 없음"
            risk_level_str = "safe"
            if display_objects:
                top = display_objects[0]
                risk_level_str = (
                    "danger" if top.risk_level == 2
                    else "warning" if top.risk_level == 1
                    else "safe"
                )
                main_hazard = build_hazard_summary([o.dict() for o in display_objects])

            safe_dir = calculate_safe_direction(analyzed_data)

            # 8. Gemini 안내 문장 생성
            guide_result = await guide_service.generate_guide(
                risk_level=risk_level_str,
                main_hazard=main_hazard,
                safe_direction=safe_dir,
                display_objects=[o.dict() for o in display_objects],
            )

            total_time = time.time() - frame_start
            print(
                f"⏱️ [WS Frame] {total_time:.3f}s | "
                f"risk={risk_level_str} | objects={len(objects)}"
            )

            # 9. JSON 결과 push
            response = PredictObjectsSpatialResponse(
                risk_level=risk_level_str,
                main_hazard=main_hazard,
                safe_direction=safe_dir,
                guide_message=guide_result.get("guide_message", ""),
                guide_source=guide_result.get("guide_source", "none"),
                process_time=f"{total_time:.3f}s",
                display_objects=display_objects,
                objects=objects,
            )
            await websocket.send_text(
                json.dumps(response.dict(), ensure_ascii=False)
            )

    except WebSocketDisconnect:
        print("🔌 [WebSocket] 클라이언트 연결 종료")
    except Exception as exc:
        print(f"❌ [WebSocket] 오류: {exc}")
        try:
            await websocket.send_text(
                json.dumps({"error": str(exc)}, ensure_ascii=False)
            )
        except Exception:
            pass
