from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

OVERLAP_RATIO_THRESHOLD_DEFAULT = 0.3
CENTER_DANGER_THRESHOLD = 5.0
SIDE_DANGER_THRESHOLD = 10.0
STOP_THRESHOLD = 15.0


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

_LABEL_KR: Dict[str, str] = {
    # COCO 80 classes (전체 한글 매핑)
    # 사람
    "person": "사람",
    # 탈것
    "bicycle": "자전거",
    "car": "자동차",
    "motorcycle": "오토바이",
    "airplane": "비행기",
    "bus": "버스",
    "train": "기차",
    "truck": "트럭",
    "boat": "보트",
    # 도로/교통
    "traffic light": "신호등",
    "fire hydrant": "소화전",
    "stop sign": "정지 표지판",
    "parking meter": "주차 미터기",
    # 야외 시설
    "bench": "벤치",
    # 동물
    "bird": "새",
    "cat": "고양이",
    "dog": "개",
    "horse": "말",
    "sheep": "양",
    "cow": "소",
    "elephant": "코끼리",
    "bear": "곰",
    "zebra": "얼룩말",
    "giraffe": "기린",
    # 액세서리
    "backpack": "가방",
    "umbrella": "우산",
    "handbag": "핸드백",
    "tie": "넥타이",
    "suitcase": "여행가방",
    # 스포츠 용품
    "frisbee": "프리스비",
    "skis": "스키",
    "snowboard": "스노보드",
    "sports ball": "공",
    "kite": "연",
    "baseball bat": "야구 방망이",
    "baseball glove": "야구 글러브",
    "skateboard": "스케이트보드",
    "surfboard": "서프보드",
    "tennis racket": "테니스 라켓",
    # 주방/식기
    "bottle": "병",
    "wine glass": "와인잔",
    "cup": "컵",
    "fork": "포크",
    "knife": "칼",
    "spoon": "숟가락",
    "bowl": "그릇",
    # 음식
    "banana": "바나나",
    "apple": "사과",
    "sandwich": "샌드위치",
    "orange": "오렌지",
    "broccoli": "브로콜리",
    "carrot": "당근",
    "hot dog": "핫도그",
    "pizza": "피자",
    "donut": "도넛",
    "cake": "케이크",
    # 가구/실내
    "chair": "의자",
    "couch": "소파",
    "potted plant": "화분",
    "bed": "침대",
    "dining table": "식탁",
    "toilet": "변기",
    # 전자기기
    "tv": "TV",
    "laptop": "노트북",
    "mouse": "마우스",
    "remote": "리모컨",
    "keyboard": "키보드",
    "cell phone": "휴대폰",
    # 가전
    "microwave": "전자레인지",
    "oven": "오븐",
    "toaster": "토스터",
    "sink": "세면대",
    "refrigerator": "냉장고",
    # 기타 실내
    "book": "책",
    "clock": "시계",
    "vase": "꽃병",
    "scissors": "가위",
    "teddy bear": "인형",
    "hair drier": "드라이어",
    "toothbrush": "칫솔",
    # 기존 커스텀 항목 (COCO 외 추가)
    "stairs": "계단",
    "door": "문",
    "table": "테이블",
    "pole": "기둥",
    "wall": "벽",
    "curb": "턱",
    "crosswalk": "횡단보도",
    "handle": "손잡이",
}

# 객체 추적 및 분석을 위한 전역 상태 (메모리 내 캐시)
# key: "label:pos_bucket" 형태로 위치별 구분 (예: "person:center")
_object_history: Dict[str, Dict[str, Any]] = {}

def _pos_bucket(x_norm: float) -> str:
    """x 중심 좌표를 left/center/right 버킷으로 변환"""
    if x_norm < 0.33:
        return "left"
    elif x_norm <= 0.66:
        return "center"
    else:
        return "right"


def _determine_motion_state(label: str, current_dist: float, current_area: float, x_norm: float = 0.5) -> str:
    """
    bbox 면적 변화(주) + 깊이 변화(보조)로 접근 여부 판단.
    키를 label+위치버킷으로 관리해 여러 동일 객체 혼용 방지.
    """
    key = f"{label}:{_pos_bucket(x_norm)}"

    if key not in _object_history:
        _object_history[key] = {"depth": [], "area": []}

    hist = _object_history[key]
    hist["depth"].append(current_dist)
    hist["area"].append(current_area)

    # 최대 5프레임 유지
    if len(hist["depth"]) > 5:
        hist["depth"].pop(0)
        hist["area"].pop(0)

    # 최소 2프레임부터 판단 (기존 3 → 2)
    if len(hist["depth"]) < 2:
        return "stable"

    depth_diffs = [hist["depth"][i] - hist["depth"][i-1] for i in range(1, len(hist["depth"]))]
    area_diffs  = [hist["area"][i]  - hist["area"][i-1]  for i in range(1, len(hist["area"]))]

    avg_depth_diff = sum(depth_diffs) / len(depth_diffs)
    avg_area_diff  = sum(area_diffs)  / len(area_diffs)

    # 면적 증가(주 신호) OR 깊이 감소(보조 신호) → 접근으로 판단
    area_fast  = avg_area_diff > 1.5     # 면적이 빠르게 증가
    area_slow  = avg_area_diff > 0.3     # 면적이 서서히 증가
    depth_fast = avg_depth_diff < -0.15  # 깊이가 빠르게 감소
    depth_slow = avg_depth_diff < -0.05  # 깊이가 서서히 감소
    moving_away = avg_area_diff < -0.3 and avg_depth_diff > 0.05

    if area_fast or (area_slow and depth_fast):
        return "approaching_fast"
    elif area_slow or depth_slow:
        return "approaching_slow"
    elif moving_away:
        return "moving_away"
    else:
        return "stable"

def _get_smoothed_distance(label: str, current_dist: float, x_norm: float = 0.5) -> float:
    """이동평균을 활용한 거리 스무딩"""
    key = f"{label}:{_pos_bucket(x_norm)}"
    if key not in _object_history:
        return current_dist

    depths = _object_history[key]["depth"]
    if not depths:
        return current_dist

    recent = depths[-3:] if len(depths) >= 3 else depths
    return sum(recent) / len(recent)

def analyze_spatial_results(
    detections: List[Any],
    depth_map: np.ndarray,
    image_width: int,
    image_height: int,
    overlap_threshold: float = 0.3,
    danger_threshold: float = 1.5,
    reference_depth: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    탐지 리스트와 깊이 맵으로 위치·거리·위험도를 계산합니다. (Phase 5.6 개선)
    """
    logger.info("spatial_analysis: 인식된 객체 수: %s", len(detections))
    if not detections:
        return []

    analyzed_objects: List[Dict[str, Any]] = []
    
    # 1. 객체별 상세 분석 수행
    chairs = [d for d in detections if _norm_label(d.label) in ("chair", "toilet", "bench", "couch", "sofa")]
    people = [d for d in detections if _norm_label(d.label) == "person"]
    others = [d for d in detections if _norm_label(d.label) not in ("chair", "person")]

    # 의자 점유 판별 및 처리
    for chair in chairs:
        is_empty = True
        for person in people:
            if _is_person_occupying_chair(chair, person, depth_map, overlap_threshold):
                is_empty = False
                break
        analyzed_objects.append(_process_single_object(chair, depth_map, image_width, image_height, is_empty=is_empty, danger_threshold=danger_threshold, reference_depth=reference_depth))

    for obj in people + others:
        analyzed_objects.append(_process_single_object(obj, depth_map, image_width, image_height, is_empty=None, danger_threshold=danger_threshold, reference_depth=reference_depth))

    # 2. 우선순위 정렬 및 display_objects 선정 (Phase 5.6 요구사항)
    # 정렬 기준: risk_level > front 우선 > bbox 면적 > 거리
    def sort_key(x):
        rl = x.get("risk_level", 0)
        pos = x.get("position", "")
        # 전방 객체에 가점
        pos_score = 10 if pos in ("front", "center") else 0
        area = x.get("area_ratio_percent", 0.0)
        dist = -x.get("estimated_distance_m", 99.0) # 가까운 것 우선
        return (rl, pos_score, area, dist)

    analyzed_objects.sort(key=sort_key, reverse=True)
    
    # 3. 보조 위험 객체 후보 선정 (회피 방향 등에 영향 주는 객체)
    # 이 부분은 calculate_safe_direction 이후에 추가 필터링이 필요할 수 있음
    return analyzed_objects


def calculate_safe_direction(analyzed_objects: List[Dict[str, Any]]) -> str:
    """
    바운딩 박스의 좌/중/우 영역 겹침 비율로 점수를 분배하여 가장 안전한 방향을 추천합니다.
    (개선) 중심점 기반 → bbox 전체 범위 기반으로 오탐 방지.
    """
    LEFT_END = 0.33
    RIGHT_START = 0.66

    left_score = 0.0
    center_score = 0.0
    right_score = 0.0

    for obj in analyzed_objects:
        rl = obj.get("risk_level", 0)
        if rl == 0:
            continue

        dist = obj.get("estimated_distance_m", 5.0)
        area = obj.get("area_ratio_percent", 1.0)
        weight = 10.0 if rl == 2 else 3.0
        obj_score = weight * (2.0 / max(dist, 0.5)) * (area / 10.0)

        # bbox 전체 범위로 좌/중/우 겹침 비율 계산
        zone_overlap = obj.get("zone_overlap")
        if zone_overlap:
            left_score   += obj_score * zone_overlap.get("left", 0.0)
            center_score += obj_score * zone_overlap.get("center", 0.0)
            right_score  += obj_score * zone_overlap.get("right", 0.0)
        else:
            # fallback: 기존 position 방식
            pos = obj.get("position", "center")
            if pos == "left":   left_score += obj_score
            elif pos == "right": right_score += obj_score
            else:                center_score += obj_score

    if center_score > CENTER_DANGER_THRESHOLD:
        if left_score < CENTER_DANGER_THRESHOLD and left_score <= right_score: return "left"
        elif right_score < CENTER_DANGER_THRESHOLD and right_score <= left_score: return "right"
        else: return "stop"

    if left_score > SIDE_DANGER_THRESHOLD and right_score < CENTER_DANGER_THRESHOLD: return "right"
    if right_score > SIDE_DANGER_THRESHOLD and left_score < CENTER_DANGER_THRESHOLD: return "left"
    if (left_score + center_score + right_score) > STOP_THRESHOLD: return "stop"

    return "forward"


def _process_single_object(
    obj: Any,
    depth_map: np.ndarray,
    width: int,
    image_height: int,
    is_empty: Optional[bool] = None,
    *,
    danger_threshold: float = 1.5,
    reference_depth: float = 1.0,
) -> Dict[str, Any]:
    x_center = int((obj.x1 + obj.x2) / 2)
    y_center = int((obj.y1 + obj.y2) / 2)
    h, w = depth_map.shape[0], depth_map.shape[1]
    xi = min(max(x_center, 0), w - 1)
    yi = min(max(y_center, 0), h - 1)

    raw_depth = float(depth_map[yi, xi])

    iw = float(width)
    ih = float(image_height)
    x_norm = x_center / iw if iw > 0 else 0.5
    y_norm = y_center / ih if ih > 0 else 0.5

    if y_norm > 0.5 and 0.33 <= x_norm <= 0.66:
        position, position_ko = "front", "전방"
    elif 0.33 <= x_norm <= 0.66:
        position, position_ko = "center", "중앙"
    elif x_norm < 0.33:
        position, position_ko = "left", "왼쪽"
    else:
        position, position_ko = "right", "오른쪽"

    lk = _norm_label(obj.label)
    label_ko = _LABEL_KR.get(lk, str(obj.label))

    # BBox 면적 계산
    x1, y1, x2, y2 = float(obj.x1), float(obj.y1), float(obj.x2), float(obj.y2)
    bbox_area_ratio = ((x2 - x1) * (y2 - y1)) / (iw * ih) * 100

    # 거리 추정 및 스무딩
    smoothed_dist = _get_smoothed_distance(lk, raw_depth, x_norm)
    motion_state = _determine_motion_state(lk, raw_depth, bbox_area_ratio, x_norm)

    # 신뢰도 판단
    confidence = "high"
    if raw_depth < 0.1 or raw_depth > 10.0:
        confidence = "low"
    elif abs(raw_depth - smoothed_dist) > 0.5:
        confidence = "medium"

    distance_text = f"약 {smoothed_dist:.1f}m"
    if confidence == "low":
        distance_text = "가까운 위치" if raw_depth < 1.0 else "멀지 않은 위치"

    description = f"{position_ko} {distance_text} 거리에 {label_ko}{_subject_particle_iga(label_ko)} 있습니다."

    risk_level = 0
    warning_threshold = danger_threshold * 1.5
    if bbox_area_ratio > 5.0 and y_norm >= 0.20:
        if smoothed_dist <= danger_threshold: risk_level = 2
        elif smoothed_dist <= warning_threshold: risk_level = 1
            
    risk_level_str = {2: "danger", 1: "warning", 0: "safe"}[risk_level]

    # 바운딩 박스가 좌/중/우 영역에 얼마나 겹치는지 비율 계산
    LEFT_END = 0.33
    RIGHT_START = 0.66
    x1_norm = x1 / iw if iw > 0 else 0.0
    x2_norm = x2 / iw if iw > 0 else 1.0
    bbox_w = max(x2_norm - x1_norm, 1e-6)
    left_overlap   = max(0.0, min(x2_norm, LEFT_END)   - max(x1_norm, 0.0))      / bbox_w
    center_overlap = max(0.0, min(x2_norm, RIGHT_START) - max(x1_norm, LEFT_END)) / bbox_w
    right_overlap  = max(0.0, min(x2_norm, 1.0)         - max(x1_norm, RIGHT_START)) / bbox_w

    return {
        "label": obj.label,
        "label_ko": label_ko,
        "confidence": round(float(obj.confidence), 4),
        "position": position,
        "position_ko": position_ko,
        "distance": f"{raw_depth:.1f}m",
        "distance_text": distance_text,
        "estimated_distance_m": round(smoothed_dist, 2),
        "raw_depth_value": round(raw_depth, 3),
        "reference_depth": reference_depth,
        "distance_confidence": confidence,
        "motion_state": motion_state,
        "is_empty": is_empty,
        "description": description,
        "x1": round(x1_norm, 4),
        "y1": round(float(y1 / ih), 4) if ih > 0 else 0,
        "x2": round(x2_norm, 4),
        "y2": round(float(y2 / ih), 4) if ih > 0 else 0,
        "risk_level": risk_level,
        "risk_level_str": risk_level_str,
        "area_ratio_percent": round(bbox_area_ratio, 2),
        "zone_overlap": {
            "left":   round(left_overlap, 4),
            "center": round(center_overlap, 4),
            "right":  round(right_overlap, 4),
        },
    }

def _norm_label(label: str) -> str:
    return str(label).lower()
