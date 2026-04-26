from io import BytesIO
import os
import sys
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
import numpy as np
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="ChikChik-Bopok Object Detection API")

# 가장 가벼운 YOLO11 가중치 사용 (없으면 ultralytics가 자동 다운로드)
MODEL_WEIGHTS = "yolo11n.pt"
model = YOLO(MODEL_WEIGHTS)

LITEMONO_REPO_PATH = os.getenv("LITEMONO_REPO_PATH", "").strip()
LITEMONO_WEIGHTS_DIR = os.getenv("LITEMONO_WEIGHTS_DIR", "").strip()
LITEMONO_MODEL_NAME = os.getenv("LITEMONO_MODEL_NAME", "lite-mono").strip()

# Lite-Mono는 disparity를 예측하므로, 상대 거리용으로 역수를 사용한다.
MIN_DEPTH = 0.1
MAX_DEPTH = 100.0


def _disp_to_depth(disp: Any, min_depth: float = MIN_DEPTH, max_depth: float = MAX_DEPTH) -> Any:
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


class LiteMonoDepthEstimator:
    def __init__(self) -> None:
        self._loaded = False
        self._error_message = ""
        self.device = None
        self.encoder = None
        self.depth_decoder = None
        self.feed_width = 640
        self.feed_height = 192

    def _load(self) -> None:
        if self._loaded:
            return

        if not LITEMONO_REPO_PATH or not os.path.isdir(LITEMONO_REPO_PATH):
            self._error_message = (
                "Lite-Mono 저장소 경로가 필요합니다. "
                "환경변수 LITEMONO_REPO_PATH에 로컬 Lite-Mono 경로를 설정하세요."
            )
            return

        if not LITEMONO_WEIGHTS_DIR or not os.path.isdir(LITEMONO_WEIGHTS_DIR):
            self._error_message = (
                "Lite-Mono 가중치 폴더가 필요합니다. "
                "환경변수 LITEMONO_WEIGHTS_DIR에 encoder.pth/depth.pth 경로를 설정하세요."
            )
            return

        encoder_path = os.path.join(LITEMONO_WEIGHTS_DIR, "encoder.pth")
        decoder_path = os.path.join(LITEMONO_WEIGHTS_DIR, "depth.pth")
        if not os.path.isfile(encoder_path) or not os.path.isfile(decoder_path):
            self._error_message = "Lite-Mono 가중치 파일 encoder.pth, depth.pth를 찾을 수 없습니다."
            return

        try:
            import torch
            from torchvision import transforms

            if LITEMONO_REPO_PATH not in sys.path:
                sys.path.append(LITEMONO_REPO_PATH)
            import networks  # type: ignore

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            encoder_dict = torch.load(encoder_path, map_location=self.device)
            decoder_dict = torch.load(decoder_path, map_location=self.device)
            self.feed_height = int(encoder_dict.get("height", 192))
            self.feed_width = int(encoder_dict.get("width", 640))

            encoder = networks.LiteMono(
                model=LITEMONO_MODEL_NAME,
                height=self.feed_height,
                width=self.feed_width,
            )
            encoder_state = encoder.state_dict()
            encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder_state})
            encoder.to(self.device)
            encoder.eval()

            depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
            decoder_state = depth_decoder.state_dict()
            depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in decoder_state})
            depth_decoder.to(self.device)
            depth_decoder.eval()

            self.encoder = encoder
            self.depth_decoder = depth_decoder
            self.to_tensor = transforms.ToTensor()
            self._loaded = True
            self._error_message = ""
        except Exception as exc:
            self._error_message = f"Lite-Mono 로딩 실패: {exc}"

    @property
    def is_ready(self) -> bool:
        self._load()
        return self._loaded

    @property
    def error_message(self) -> str:
        self._load()
        return self._error_message

    def predict_depth_map(self, pil_image: Image.Image) -> np.ndarray:
        self._load()
        if not self._loaded:
            raise RuntimeError(self._error_message or "Lite-Mono가 준비되지 않았습니다.")

        import torch

        original_width, original_height = pil_image.size
        resized = pil_image.resize((self.feed_width, self.feed_height), Image.LANCZOS)
        tensor = self.to_tensor(resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.encoder(tensor)
            outputs = self.depth_decoder(features)
            disp = outputs[("disp", 0)]
            _, depth = _disp_to_depth(disp, MIN_DEPTH, MAX_DEPTH)
            depth_resized = torch.nn.functional.interpolate(
                depth,
                (original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )

        depth_map = depth_resized.squeeze().detach().cpu().numpy().astype(np.float32)
        return depth_map


depth_estimator = LiteMonoDepthEstimator()


def _extract_objects(result: Any, image_area: float, depth_map: np.ndarray | None = None) -> List[Dict[str, Any]]:
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

        obj: Dict[str, Any] = {
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
                obj["is_dangerous"] = bool(obj["is_over_30_percent"] or distance_level == "near")
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


@app.post("/predict/objects")
async def predict_objects(file: UploadFile = File(...)) -> Dict[str, Any]:
    pil_image = await _read_image(file)

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
    detected_objects = _extract_objects(result, image_area=image_area)

    return {
        "filename": file.filename,
        "image_size": {"width": image_width, "height": image_height},
        "objects": detected_objects,
        "total_objects": len(detected_objects),
    }


@app.post("/predict/objects-distance")
async def predict_objects_with_distance(file: UploadFile = File(...)) -> Dict[str, Any]:
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
        return {
            "filename": file.filename,
            "image_size": {"width": image_width, "height": image_height},
            "objects": [],
            "total_objects": 0,
            "distance_model": LITEMONO_MODEL_NAME,
        }

    depth_map = depth_estimator.predict_depth_map(pil_image)
    detected_objects = _extract_objects(results[0], image_area=image_area, depth_map=depth_map)

    return {
        "filename": file.filename,
        "image_size": {"width": image_width, "height": image_height},
        "objects": detected_objects,
        "total_objects": len(detected_objects),
        "distance_model": LITEMONO_MODEL_NAME,
    }
