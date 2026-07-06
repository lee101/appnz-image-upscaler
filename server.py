import base64
import os
import threading
import urllib.request

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

PORT = int(os.environ.get("PORT", "5000"))
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
TILE = int(os.environ.get("TILE", "512"))
state = {"loaded": False, "error": ""}
UPSAMPLER = None
FACE = {}
LOCK = threading.Lock()

COG_SCHEMA = {
    "outputKind": "image",
    "inputs": [
        {
            "name": "image",
            "type": "image",
            "description": "Image (https URL or data URI)",
            "required": True,
            "order": 0,
        },
        {
            "name": "scale",
            "type": "integer",
            "description": "Upscale factor",
            "default": 4,
            "choices": [2, 4],
            "required": False,
            "order": 1,
        },
        {
            "name": "face_enhance",
            "type": "boolean",
            "description": "Restore faces with GFPGAN",
            "default": False,
            "required": False,
            "order": 2,
        },
    ],
}


def _get_face(scale):
    if scale not in FACE:
        from gfpgan import GFPGANer

        FACE[scale] = GFPGANer(
            model_path=os.path.join(MODEL_DIR, "GFPGANv1.4.pth"),
            upscale=scale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=UPSAMPLER,
        )
    return FACE[scale]


def _load():
    global UPSAMPLER
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
        )
        UPSAMPLER = RealESRGANer(
            scale=4,
            model_path=os.path.join(MODEL_DIR, "RealESRGAN_x4plus.pth"),
            model=model,
            tile=TILE,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
        )
        _get_face(4)
        state["loaded"] = True
    except Exception as e:
        state["error"] = str(e)


def fetch_bytes(v):
    if not isinstance(v, str) or not v:
        raise HTTPException(status_code=400, detail="missing image")
    if v.startswith("data:"):
        return base64.b64decode(v.partition(",")[2])
    if v.startswith(("http://", "https://")):
        with urllib.request.urlopen(v, timeout=120) as r:
            return r.read()
    return base64.b64decode(v)


def upscale(data, scale, face_enhance):
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="could not decode image")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if face_enhance:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        _, _, out = _get_face(scale).enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )
    else:
        out, _ = UPSAMPLER.enhance(img, outscale=scale)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise HTTPException(status_code=500, detail="png encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


app = FastAPI(title="appnz-image-upscaler", openapi_url=None)


@app.on_event("startup")
def startup():
    threading.Thread(target=_load, daemon=True).start()


@app.get("/health-check")
def health_check():
    if state["error"]:
        return {"status": "SETUP", "error": state["error"]}
    return {"status": "READY" if state["loaded"] else "SETUP"}


@app.get("/healthz")
def healthz():
    if not state["loaded"]:
        return JSONResponse({"status": "loading", "error": state["error"]}, status_code=503)
    return {"status": "ok"}


@app.get("/openapi.json")
def openapi_json():
    return COG_SCHEMA


@app.post("/predictions")
def predictions(payload: dict):
    if not state["loaded"]:
        return JSONResponse(
            {"status": "failed", "error": state["error"] or "model loading"},
            status_code=503,
        )
    body = payload.get("input") or {}
    try:
        scale = int(body.get("scale") or 4)
    except (TypeError, ValueError):
        return JSONResponse({"status": "failed", "error": "bad scale"}, status_code=400)
    if scale not in (2, 4):
        return JSONResponse({"status": "failed", "error": "scale must be 2 or 4"}, status_code=400)
    face_enhance = bool(body.get("face_enhance"))
    try:
        data = fetch_bytes(body.get("image"))
        with LOCK:
            out = upscale(data, scale, face_enhance)
        return {"status": "succeeded", "output": out}
    except HTTPException as e:
        return JSONResponse({"status": "failed", "error": e.detail}, status_code=e.status_code)
    except Exception as e:
        return JSONResponse({"status": "failed", "error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
