import os
import time
import asyncio
import google.generativeai as genai
from typing import List, Dict, Any, Optional


def _wa_gwa(word: str) -> str:
    """마지막 글자 받침 여부로 '와'/'과' 선택"""
    if not word:
        return "와"
    last = word[-1]
    if not ('가' <= last <= '힣'):
        return "와"
    return "과" if (ord(last) - ord('가')) % 28 != 0 else "와"


def _count_unit(label_ko: str) -> str:
    """사람→명, 동물→마리, 나머지→개"""
    if label_ko == "사람":
        return "명"
    if label_ko in ("개", "고양이", "새", "말", "소", "양", "곰", "코끼리", "기린", "얼룩말"):
        return "마리"
    return "개"


def build_hazard_summary(display_objects: List[Dict[str, Any]]) -> str:
    """객체 목록을 그룹화해 '전방 사람 2명과 노트북 주의' 형태로 반환"""
    counts: Dict[tuple, int] = {}
    order = []
    for obj in display_objects:
        key = (obj.get('position_ko', '전방'), obj.get('label_ko', '물체'))
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += 1

    parts = []
    for pos, label in order:
        count = counts[(pos, label)]
        if count > 1:
            parts.append(f"{pos} {label} {count}{_count_unit(label)}")
        else:
            parts.append(f"{pos} {label}")

    if not parts:
        return "알 수 없는 위험"
    if len(parts) == 1:
        return parts[0]
    connector = _wa_gwa(parts[-2])
    if len(parts) == 2:
        return f"{parts[0]}{connector} {parts[1]}"
    return ", ".join(parts[:-1]) + f"{connector} {parts[-1]}"

class GuideService:
    def __init__(self):
        # 환경변수에서 API 키 및 모델명 로드
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
        self.model = None
        self._cache = {}
        
        # Circuit Breaker 상태 변수
        self.gemini_disabled_until = 0.0  # 타임스탬프
        self.last_quota_error_log_time = 0.0
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # 모델명 앞에 'models/'를 붙이지 않고 SDK 표준 형식으로 사용
                self.model = genai.GenerativeModel(self.model_name)
                print(f"✅ [GuideService] Gemini API ({self.model_name}) 구성 완료.")
            except Exception as e:
                print(f"❌ [GuideService] Gemini 구성 실패: {e}")
        else:
            print("⚠️ [GuideService] GEMINI_API_KEY를 찾을 수 없습니다. Rule-based 모드로 동작합니다.")

    async def generate_guide(
        self, 
        risk_level: str, 
        main_hazard: str, 
        safe_direction: str, 
        display_objects: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Gemini를 호출하여 자연스러운 안내 문장을 생성합니다. (Phase 5.6 개선)
        """
        start_time = time.time()
        
        # 디버그 로그 출력 (Phase 5.6 요구사항)
        print(f"🔍 [GuideDebug] display_objects count={len(display_objects)}")
        for i, obj in enumerate(display_objects):
            label = obj.get('label_ko', 'unknown')
            pos = obj.get('position_ko', 'unknown')
            risk = obj.get('risk_level_str', 'safe')
            dist = obj.get('distance_text', 'unknown')
            motion = obj.get('motion_state', 'stable')
            print(f"🔍 [GuideDebug] obj{i+1}={label}/{pos}/{risk}/{dist}/{motion}")

        # 0. Circuit Breaker 확인
        current_time = time.time()
        if self.gemini_disabled_until > current_time:
            return self._fallback(risk_level, main_hazard, safe_direction, display_objects, "quota_disabled", "rule_based_quota_fallback")

        # 1. 접근 중인 객체가 있으면 캐시 무시 (실시간 우선)
        motion_states = [o.get('motion_state', 'stable') for o in display_objects]
        has_approaching = any(m in ('approaching_fast', 'approaching_slow') for m in motion_states)

        # 캐시 키에 motion 상태 포함
        motion_summary = "_".join(sorted(set(motion_states)))
        cache_key = f"{risk_level}_{main_hazard}_{safe_direction}_{len(display_objects)}_{motion_summary}"

        if not has_approaching and cache_key in self._cache:
            return {
                "guide_message": self._cache[cache_key],
                "guide_source": "gemini",
                "cache_hit": "true",
                "process_time": f"{(time.time() - start_time):.3f}s"
            }

        # 2. Gemini 사용 불가 시 Fallback
        if not self.model:
            return self._fallback(risk_level, main_hazard, safe_direction, display_objects, "no_key", "rule_based")

        # 3. 프롬프트 생성
        prompt = self._build_prompt(risk_level, main_hazard, safe_direction, display_objects)
        
        try:
            # 4. Gemini 호출
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.model.generate_content(prompt)),
                timeout=1.5
            )
            
            if response and response.text:
                guide_msg = response.text.strip().replace('"', '').replace("'", "")
                if len(guide_msg) > 70: guide_msg = guide_msg[:70]
                
                self._cache[cache_key] = guide_msg
                elapsed = time.time() - start_time
                print(f"🚀 [GuideService] Gemini 생성 성공 ({elapsed:.3f}s): {guide_msg}")
                
                return {
                    "guide_message": guide_msg,
                    "guide_source": "gemini",
                    "cache_hit": "false",
                    "process_time": f"{elapsed:.3f}s"
                }
            else:
                return self._fallback(risk_level, main_hazard, safe_direction, display_objects, "empty_response", "rule_based_error_fallback")
                
        except Exception as e:
            err_msg = str(e)
            if any(k in err_msg for k in ["429", "quota", "rate limit", "ResourceExhausted"]):
                self.gemini_disabled_until = time.time() + 60.0
                return self._fallback(risk_level, main_hazard, safe_direction, display_objects, "quota_exceeded", "rule_based_quota_fallback")
            return self._fallback(risk_level, main_hazard, safe_direction, display_objects, err_msg, "rule_based_error_fallback")

    def _build_prompt(self, risk_level: str, main_hazard: str, safe_direction: str, display_objects: List[Dict[str, Any]]) -> str:
        """Gemini용 시스템 프롬프트 및 컨텍스트 생성"""
        hazard_summary = build_hazard_summary(display_objects)

        objects_summary = ""
        for obj in display_objects:
            label = obj.get('label_ko', 'unknown')
            pos = obj.get('position_ko', 'center')
            dist = obj.get('distance_text', 'unknown')
            motion = obj.get('motion_state', 'stable')
            conf = obj.get('distance_confidence', 'medium')
            motion_desc = ""
            if motion == "approaching_fast": motion_desc = " (빠르게 접근 중)"
            elif motion == "approaching_slow": motion_desc = " (서서히 접근 중)"
            objects_summary += f"- {label}: {pos}, {dist}{motion_desc}, 거리신뢰도: {conf}\n"

        return f"""당신은 시각장애인을 위한 실시간 AI 보행 보조 시스템 Vision Aid AI입니다.
사용자가 즉시 행동할 수 있는 짧고 명확한 한국어 안내 문장을 생성하세요.

[감지된 위험 요약]: {hazard_summary}
[위험 등급]: {risk_level}
[추천 안전 방향]: {safe_direction}
[감지 상세]:
{objects_summary}

[지침]
1. 절대 이미지 설명 금지 ("보입니다", "사진 속에" 등 금지)
2. 같은 종류 객체가 여러 명/개면 숫자를 포함하세요. 예: "전방 사람 2명과 노트북 주의"
3. 거리 신뢰도 'low'면 숫자 대신 "가까운 위치"로 표현하세요.
4. 빠르게 접근 중(approaching_fast)이면 최우선으로 "멈추세요" 안내.
5. 서서히 접근 중(approaching_slow)이면 "가까워지고 있습니다"와 방향 안내를 포함하세요.
6. 추천 방향({safe_direction})을 행동으로 지시하세요.
7. 1~2문장, 짧고 행동 중심적으로 작성하세요.
7. 출력은 오직 한국어 안내 문장만 하세요.
"""

    def _fallback(self, risk_level: str, main_hazard: str, safe_direction: str, display_objects: List[Dict[str, Any]], reason: str, source: str = "rule_based") -> Dict[str, str]:
        """Gemini 실패 시 Rule-based 안내"""
        start_time = time.time()

        dir_map = {"left": "왼쪽으로 이동하세요.", "right": "오른쪽으로 이동하세요.", "stop": "잠시 멈추세요.", "forward": "천천히 직진하세요."}
        action = dir_map.get(safe_direction, "주의하세요.")

        if display_objects:
            # 1순위: 빠르게 접근
            fast = next((o for o in display_objects if o.get('motion_state') == 'approaching_fast'), None)
            # 2순위: 서서히 접근
            slow = next((o for o in display_objects if o.get('motion_state') == 'approaching_slow'), None)

            if fast:
                msg = f"{fast['position_ko']} {fast['label_ko']}가 빠르게 가까워집니다. {action}"
            elif slow:
                msg = f"{slow['position_ko']} {slow['label_ko']}가 가까워지고 있습니다. {action}"
            else:
                summary = build_hazard_summary(display_objects)
                msg = f"{summary} 주의. {action}"
        else:
            msg = f"주의. {main_hazard}. {action}" if main_hazard != "감지된 위험 요소 없음" else action

        elapsed = time.time() - start_time
        return {
            "guide_message": msg,
            "guide_source": source,
            "fallback_reason": reason,
            "process_time": f"{elapsed:.3f}s"
        }

# 싱글톤 인스턴스
guide_service = GuideService()

# 싱글톤 인스턴스
guide_service = GuideService()
