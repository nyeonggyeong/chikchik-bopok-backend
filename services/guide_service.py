import os
import time
import asyncio
import google.generativeai as genai
from typing import List, Dict, Any, Optional

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

        # 1. 캐시 확인
        cache_key = f"{risk_level}_{main_hazard}_{safe_direction}_{len(display_objects)}"
        if cache_key in self._cache:
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
        """Gemini용 시스템 프롬프트 및 컨텍스트 생성 (Phase 5.6)"""
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
당신의 역할은 감지된 위험 객체 정보(최대 2개)와 안전 방향을 바탕으로 사용자가 즉시 행동할 수 있는 짧고 명확한 한국어 안내 문장을 생성하는 것입니다.

[상황 데이터]
- 위험 등급: {risk_level}
- 추천 안전 방향: {safe_direction}
- 감지된 주요 물체들:
{objects_summary}

[지침]
1. 절대 이미지 설명을 하지 마세요. ("보입니다", "사진 속에" 등 금지)
2. 가장 위험한 객체를 먼저 언급하고, 두 번째 객체가 위험하거나 이동 경로에 있으면 반드시 포함하세요.
3. 거리 신뢰도가 'low'인 경우 숫자 대신 "가까운 위치" 등으로 표현하세요.
4. 물체가 빠르게 접근 중(approaching_fast)이면 최우선으로 "멈추세요"라고 안내하세요.
5. 추천 안전 방향({safe_direction})을 행동으로 지시하세요:
   - "전방 의자를 피해 오른쪽으로 이동하세요."
   - "전방 의자와 오른쪽 사람 주의. 왼쪽으로 이동하세요."
6. 문장은 1~2문장으로 아주 짧고 행동 중심적으로 작성하세요.
7. 출력은 오직 한국어 안내 문장만 하세요.
"""

    def _fallback(self, risk_level: str, main_hazard: str, safe_direction: str, display_objects: List[Dict[str, Any]], reason: str, source: str = "rule_based") -> Dict[str, str]:
        """Gemini 실패 시 사용할 고품질 Rule-based 안내 (Phase 5.6)"""
        start_time = time.time()
        
        dir_map = {"left": "왼쪽으로 이동하세요.", "right": "오른쪽으로 이동하세요.", "stop": "잠시 멈추세요.", "forward": "천천히 직진하세요."}
        action = dir_map.get(safe_direction, "주의하세요.")

        msg = ""
        if len(display_objects) >= 2:
            obj1 = display_objects[0]
            obj2 = display_objects[1]
            # 두 객체가 모두 위험한 경우
            msg = f"{obj1['position_ko']} {obj1['label_ko']}와 {obj2['position_ko']} {obj2['label_ko']} 주의. {action}"
        elif len(display_objects) == 1:
            obj = display_objects[0]
            if obj['motion_state'] == 'approaching_fast':
                msg = f"{obj['position_ko']} {obj['label_ko']}가 빠르게 가까워집니다. {action}"
            else:
                msg = f"{obj['position_ko']} {obj['distance_text']} 지점 {obj['label_ko']} 주의. {action}"
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
