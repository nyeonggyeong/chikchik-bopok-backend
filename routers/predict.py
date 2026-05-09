from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile

from services.depth import LITEMONO_MODEL_NAME, depth_estimator
from services.detection import _extract_objects, _read_image, model

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/objects")
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
    detected_objects = _extract_objects(result, image_width=float(image_width), image_height=float(image_height))

    return {
        "filename": file.filename,
        "image_size": {"width": image_width, "height": image_height},
        "objects": detected_objects,
        "total_objects": len(detected_objects),
    }


@router.post("/objects-distance")
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
    detected_objects = _extract_objects(results[0], image_width=float(image_width), image_height=float(image_height), depth_map=depth_map)

    return {
        "filename": file.filename,
        "image_size": {"width": image_width, "height": image_height},
        "objects": detected_objects,
        "total_objects": len(detected_objects),
        "distance_model": LITEMONO_MODEL_NAME,
    }
