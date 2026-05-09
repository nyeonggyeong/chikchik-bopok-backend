import os
import sys
from typing import Any

import numpy as np
from PIL import Image

LITEMONO_REPO_PATH = os.getenv("LITEMONO_REPO_PATH", "").strip()
LITEMONO_WEIGHTS_DIR = os.getenv("LITEMONO_WEIGHTS_DIR", "").strip()
LITEMONO_MODEL_NAME = os.getenv("LITEMONO_MODEL_NAME", "lite-mono").strip()

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


# 싱글톤 인스턴스 — 앱 전체에서 공유
depth_estimator = LiteMonoDepthEstimator()
