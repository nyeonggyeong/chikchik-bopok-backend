# chikchik-bopok-server

FastAPI backend for object detection using YOLO11 (`yolo11n.pt`) and optional Lite-Mono distance estimation.

## Features

- `POST /predict/objects` image upload endpoint
- `POST /predict/objects-distance` object + distance endpoint
- `GET /health` model readiness check endpoint
- Object detection with Ultralytics YOLO
- Distance estimation with Lite-Mono (optional, local weights)
- Returns:
  - `class_name`
  - `confidence`
  - `bbox` (`x`, `y`, `w`, `h`)
  - `area_ratio_percent`
  - `is_over_30_percent`
  - `distance_estimate_m` (only `/predict/objects-distance`)
  - `distance_level` (near/mid/far, only `/predict/objects-distance`)
  - `is_dangerous` (only `/predict/objects-distance`)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

## Lite-Mono Setup (Optional)

1. Clone Lite-Mono locally:

```bash
git clone https://github.com/noahzn/Lite-Mono.git
```

2. Download Lite-Mono weights folder containing:
- `encoder.pth`
- `depth.pth`

3. Set environment variables before running server:

```bash
set LITEMONO_REPO_PATH=C:\path\to\Lite-Mono
set LITEMONO_WEIGHTS_DIR=C:\path\to\lite-mono-weights
set LITEMONO_MODEL_NAME=lite-mono
```

If these are not configured, `/predict/objects-distance` returns `503`.

## API Test

Open:

- [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

Check model status:

```bash
curl "http://127.0.0.1:8001/health"
```

Or send multipart request:

```bash
curl -X POST "http://127.0.0.1:8001/predict/objects" -F "file=@test.jpg"
```

```bash
curl -X POST "http://127.0.0.1:8001/predict/objects-distance" -F "file=@test.jpg"
```

## Notes

- `yolo11n.pt` is excluded from git and downloaded automatically if missing.
- For mobile device testing, use your PC local IP (same Wi-Fi), not `127.0.0.1`.
