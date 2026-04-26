# chikchik-bopok-server

FastAPI backend for object detection using YOLO11 (`yolo11n.pt`).

## Features

- `POST /predict/objects` image upload endpoint
- Object detection with Ultralytics YOLO
- Returns:
  - `class_name`
  - `confidence`
  - `bbox` (`x`, `y`, `w`, `h`)
  - `area_ratio_percent`
  - `is_over_30_percent`

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

## API Test

Open:

- [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

Or send multipart request:

```bash
curl -X POST "http://127.0.0.1:8001/predict/objects" -F "file=@test.jpg"
```

## Notes

- `yolo11n.pt` is excluded from git and downloaded automatically if missing.
- For mobile device testing, use your PC local IP (same Wi-Fi), not `127.0.0.1`.
