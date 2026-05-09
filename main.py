from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, Form

# 환경 변수 로드가 최우선
load_dotenv()

from routers import predict
from services.depth import LITEMONO_MODEL_NAME, depth_estimator
from services.detection import MODEL_WEIGHTS

app = FastAPI(title="ChikChik-Bopok Object Detection API")

# Router 등록
app.include_router(predict.router)


@app.get("/health")
async def health() -> Dict[str, Any]:
    lite_mono_ready = depth_estimator.is_ready
    return {
        "status": "ok",
        "models": {
            "yolo": {"ready": True, "weights": MODEL_WEIGHTS},
            "lite_mono": {
                "ready": lite_mono_ready,
                "model_name": LITEMONO_MODEL_NAME,
                "error": None if lite_mono_ready else depth_estimator.error_message,
            },
        },
    }