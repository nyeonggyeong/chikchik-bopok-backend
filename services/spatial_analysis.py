from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

OVERLAP_RATIO_THRESHOLD_DEFAULT = 0.3


def _subject_particle_iga(word: str) -> str:
    """
    종성(받침) 여부로 주격 조사 선택: 받침 있음 → '이', 없음 → '가'.
    영문 등 비한글 문자열에는 '가'를 사용합니다.
    """
    if not word:
        return "가"
    last = word[-1]
    if not ("가" <= last <= "힣"):
        return "가"
    code = ord(last) - ord("가")
    has_batchim = (code % 28) != 0
    return "이" if has_batchim else "가"


def _describe_with_iga(speech_pos: str, dist_val: float, noun_phrase: str) -> str:
    """'{위치} {거리}미터 지점에 {명사}+이/가 있습니다.'"""
    np_ = noun_phrase.strip()
    pl = _subject_particle_iga(np_)
    return f"{speech_pos} {dist_val:.1f}미터 지점에 {np_}{pl} 있습니다."


def _is_person_occupying_chair(chair_obj, person_obj, depth_map, threshold=0.3):
    """
    의자에 사람이 앉아 있는지 판별하는 개선된 로직
    """
    c = (chair_obj.x1, chair_obj.y1, chair_obj.x2, chair_obj.y2)
    p = (person_obj.x1, person_obj.y1, person_obj.x2, person_obj.y2)
    
    # 1. 2D 겹침 계산 (Intersection)
    inter_x1 = max(c[0], p[0])
    inter_y1 = max(c[1], p[1])
    inter_x2 = min(c[2], p[2])
    inter_y2 = min(c[3], p[3])
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return False

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    chair_area = (c[2] - c[0]) * (c[3] - c[1])
    
    # 2. 의자 면적 대비 겹침 비율 (IoC: Intersection over Chair)
    ioc = inter_area / chair_area if chair_area > 0 else 0
    
    # 3. 거리(Depth) 기반 필터링 (선택 사항이지만 강력함)
    # 의자와 사람의 거리가 너무 다르면(예: 의자 뒤에 멀리 서 있는 사람) 무시
    # (depth_map을 활용해 각 객체의 중심점 거리를 비교)

    # 4. 판별 기준: 
    # 의자 영역의 30% 이상이 사람과 겹치고, 
    # 사람 박스의 가로 중심이 의자 박스 가로 범위 내에 대략 들어오면 '점유'로 간주
    p_center_x = (p[0] + p[2]) / 2
    x_aligned = (c[0] - 20 <= p_center_x <= c[2] + 20) # 약간의 여유값(20px) 부여

    return ioc >= threshold and x_aligned


@dataclass
class SpatialDetection:
    """YOLO 박스에서 변환한 최소 탐지 단위 (analyze_spatial_results 입력용)."""

    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


def detections_from_yolo(yolo_image_result: Any) -> List[SpatialDetection]:
    """Ultralytics 단일 이미지 결과(`results[0]`)를 `SpatialDetection` 리스트로 변환합니다."""
    if yolo_image_result is None:
        return []
    boxes = getattr(yolo_image_result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names = yolo_image_result.names
    out: List[SpatialDetection] = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0].item())
        class_id = int(box.cls[0].item())
        label = names[class_id] if class_id in names else str(class_id)
        out.append(
            SpatialDetection(
                label=str(label),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=confidence,
            )
        )

    return out


def _norm_label(label: str) -> str:
    return str(label).lower()


_LABEL_KR: Dict[str, str] = {
    "chair": "의자",
    "person": "사람",
    "door": "문",
    "handle": "손잡이",
}


def analyze_spatial_results(
    detections: List[Any],
    depth_map: np.ndarray,
    image_width: int,
    image_height: int,
    overlap_threshold: float = 0.3,
    danger_threshold: float = 1.5,
) -> List[Dict[str, Any]]:
    """
    탐지 리스트와 깊이 맵으로 위치·거리·빈 좌석·음성 설명을 계산합니다.

    빈 좌석이 아니라 판별하려면 (1) 교집합 면적/의자 면적 비율이 임계 이상이고
    (2) 사람 박스 하단 중심점(발 근처)이 의자 박스 안에 있어야 합니다.
    """
    logger.info("spatial_analysis: 인식된 객체 수: %s", len(detections))
    if not detections:
        return []

    chairs = [d for d in detections if _norm_label(d.label) in ("chair", "toilet", "bench", "couch")]
    people = [d for d in detections if _norm_label(d.label) == "person"]
    others = [d for d in detections if _norm_label(d.label) not in ("chair", "person")]

    analyzed_objects: List[Dict[str, Any]] = []

    for chair in chairs:
        is_empty = True
        # 디버깅을 위해 현재 의자 정보 로그
        logger.info(f"의자 분석 중... Conf: {chair.confidence:.2f}")

        for person in people:
            if _is_person_occupying_chair(chair, person, depth_map, overlap_threshold):
                is_empty = False
                logger.info(f" -> [점유됨] 사람 박스와 중첩 확인")
                break
        
        if is_empty:
            logger.info(f" -> [빈 좌석!] 점유자 없음")

        analyzed_objects.append(
            _process_single_object(
                chair,
                depth_map,
                image_width,
                image_height,
                is_empty=is_empty,
                danger_threshold=danger_threshold,
            )
        )

    for obj in people + others:
        analyzed_objects.append(
            _process_single_object(
                obj,
                depth_map,
                image_width,
                image_height,
                is_empty=None,
                danger_threshold=danger_threshold,
            )
        )

    priority_map = {"chair": 0, "person": 1, "handle": 2, "door": 3}
    analyzed_objects.sort(
        key=lambda x: (priority_map.get(_norm_label(x["label"]), 99), -float(x["confidence"]))
    )

    return analyzed_objects


def _process_single_object(
    obj: Any,
    depth_map: np.ndarray,
    width: int,
    image_height: int,
    is_empty: Optional[bool] = None,
    *,
    danger_threshold: float = 1.5,
) -> Dict[str, Any]:
    x_center = int((obj.x1 + obj.x2) / 2)
    y_center = int((obj.y1 + obj.y2) / 2)
    h, w = depth_map.shape[0], depth_map.shape[1]
    xi = min(max(x_center, 0), w - 1)
    yi = min(max(y_center, 0), h - 1)

    dist_val = float(depth_map[yi, xi])

    iw = float(width)
    ih = float(image_height)
    pos_ratio = (x_center / iw) if iw > 0 else 0.5

    if pos_ratio < 0.33:
        position = "왼쪽"
    elif pos_ratio > 0.66:
        position = "오른쪽"
    else:
        position = "중앙"

    speech_pos = "정면" if position == "중앙" else position
    lk = _norm_label(obj.label)
    display_kr = _LABEL_KR.get(lk, str(obj.label))

    if lk == "chair" and is_empty is True:
        description = _describe_with_iga(speech_pos, dist_val, "빈 좌석")
    elif lk == "chair" and is_empty is False:
        description = f"{speech_pos} {dist_val:.1f}미터 지점의 의자는 사용 중입니다."
    else:
        description = _describe_with_iga(speech_pos, dist_val, display_kr)

    x1, y1, x2, y2 = float(obj.x1), float(obj.y1), float(obj.x2), float(obj.y2)
    w_px = max(0.0, x2 - x1)
    h_px = max(0.0, y2 - y1)
    image_area = iw * ih
    bbox_area = w_px * h_px
    area_ratio_percent = (bbox_area / image_area) * 100 if image_area > 0 else 0.0

    distance_estimate_m = round(dist_val, 2)
    distance_level: str = (
        "near" if dist_val < 1.5 else ("mid" if dist_val < 3.0 else "far")
    )
    confidence_f = round(float(obj.confidence), 4)
    is_dangerous = bool(area_ratio_percent > 20.0 and dist_val <= danger_threshold)

    return {
    "label": obj.label,
    "confidence": confidence_f,
    "position": position,
    "distance": f"{dist_val:.1f}m",
    "is_empty": is_empty,
    "description": description,
    
    # Flutter가 '비율 * 화면크기' 연산을 수행할 수 있도록 비율로 변경합니다.
    "x1": round(float(x1 / iw if iw > 0 else 0.0), 4),
    "y1": round(float(y1 / ih if ih > 0 else 0.0), 4),
    "x2": round(float(x2 / iw if iw > 0 else 0.0), 4),
    "y2": round(float(y2 / ih if ih > 0 else 0.0), 4),
    
        "bbox_xyxy_px": {"x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2)},
        "bbox": {
            "x": round(float(x1 / iw if iw > 0 else 0.0), 4),
            "y": round(float(y1 / ih if ih > 0 else 0.0), 4),
            "w": round(float(w_px / iw if iw > 0 else 0.0), 4),
            "h": round(float(h_px / ih if ih > 0 else 0.0), 4),
        },
        "distance_estimate_m": distance_estimate_m,
        "distance_level": distance_level,
        "is_dangerous": is_dangerous,
        "area_ratio_percent": round(area_ratio_percent, 2),
        "is_over_30_percent": area_ratio_percent > 30.0,
    }
