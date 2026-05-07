"""mFlux HTTP 서버 — Flux1 모델 상주 + 아이들 타임아웃.

- 첫 요청 시 모델 로드, 이후 재사용
- 마지막 요청 후 3분 경과 시 자동 종료 (Metal GPU 메모리 OS 반환)
- 다음 요청 시 diffusionkit_service가 자동 재기동

포트: 18190
엔드포인트:
  POST /generate   — Flux1 dev txt2img / img2img
  POST /kontext    — Flux1 Kontext 멀티ref 합성 (image_b64 필수)
  GET  /health
  POST /shutdown
"""

import argparse
import base64
import io
import json
import logging
import os
import signal
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Timer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mflux-server")

_hf_home = Path.home() / "hf_home"
if _hf_home.exists() and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_hf_home)

DEFAULT_PORT = 18190
IDLE_TIMEOUT = 180  # 3분

_flux_dev = None
_flux_schnell = None
_flux_kontext = None
_loaded_dev_quantize = None
_loaded_schnell_quantize = None
_loaded_kontext_quantize = None
_lock = threading.Lock()
_idle_timer: Timer | None = None
_last_request_time = 0.0


def _schedule_idle_shutdown():
    global _idle_timer
    if _idle_timer:
        _idle_timer.cancel()
    _idle_timer = Timer(IDLE_TIMEOUT, _idle_shutdown)
    _idle_timer.daemon = True
    _idle_timer.start()


def _idle_shutdown():
    global _flux_dev, _flux_schnell, _loaded_dev_quantize, _loaded_schnell_quantize, _flux_kontext, _loaded_kontext_quantize
    elapsed = time.time() - _last_request_time
    if elapsed < IDLE_TIMEOUT - 5:
        _schedule_idle_shutdown()
        return
    logger.info(f"아이들 {IDLE_TIMEOUT}초 경과 → 프로세스 종료 (Metal GPU 메모리 반환)")
    os.kill(os.getpid(), signal.SIGTERM)


def get_flux(quantize: int = 8, model_name: str = "dev"):
    global _flux_dev, _flux_schnell, _loaded_dev_quantize, _loaded_schnell_quantize
    with _lock:
        if model_name == "schnell":
            if _flux_schnell is None or _loaded_schnell_quantize != quantize:
                logger.info(f"Flux1 schnell 모델 로딩 (quantize={quantize})...")
                t = time.time()
                from mflux.models.common.config import ModelConfig
                from mflux.models.flux.variants.txt2img.flux import Flux1
                _flux_schnell = Flux1(
                    model_config=ModelConfig.from_name(model_name="schnell"),
                    quantize=quantize,
                )
                _loaded_schnell_quantize = quantize
                logger.info(f"Flux1 schnell 로딩 완료: {time.time() - t:.1f}s")
            return _flux_schnell
        else:
            if _flux_dev is None or _loaded_dev_quantize != quantize:
                logger.info(f"Flux1 dev 모델 로딩 (quantize={quantize})...")
                t = time.time()
                from mflux.models.common.config import ModelConfig
                from mflux.models.flux.variants.txt2img.flux import Flux1
                _flux_dev = Flux1(
                    model_config=ModelConfig.from_name(model_name="dev"),
                    quantize=quantize,
                )
                _loaded_dev_quantize = quantize
                logger.info(f"Flux1 dev 로딩 완료: {time.time() - t:.1f}s")
            return _flux_dev


def get_flux_kontext(quantize: int = 8):
    global _flux_kontext, _loaded_kontext_quantize
    with _lock:
        if _flux_kontext is None or _loaded_kontext_quantize != quantize:
            logger.info(f"Flux1Kontext 모델 로딩 (quantize={quantize})...")
            t = time.time()
            from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
            _flux_kontext = Flux1Kontext(quantize=quantize)
            _loaded_kontext_quantize = quantize
            logger.info(f"Flux1Kontext 로딩 완료: {time.time() - t:.1f}s")
    return _flux_kontext


def _shutdown():
    logger.info("프로세스 종료")
    os.kill(os.getpid(), signal.SIGTERM)


def _save_tmp_png(data_uri: str) -> str:
    """data URI → 임시 PNG 파일 경로 반환."""
    raw = data_uri.split(",", 1)[-1]
    img_bytes = base64.b64decode(raw)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(img_bytes)
    tmp.close()
    return tmp.name


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.info(format % args)

    def _send_json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "engine": "mflux",
                "loaded_dev": _flux_dev is not None,
                "loaded_schnell": _flux_schnell is not None,
                "kontext_loaded": _flux_kontext is not None,
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        global _last_request_time

        if self.path == "/shutdown":
            self._send_json(200, {"success": True})
            Timer(0.3, _shutdown).start()
            return

        if self.path not in ("/generate", "/kontext"):
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except Exception:
            self._send_json(400, {"success": False, "error": "invalid JSON"})
            return

        _last_request_time = time.time()

        if self.path == "/generate":
            self._handle_generate(body)
        else:
            self._handle_kontext(body)

        _last_request_time = time.time()
        _schedule_idle_shutdown()

    def _handle_generate(self, body: dict):
        prompt = body.get("prompt", "")
        width = int(body.get("width", 720))
        height = int(body.get("height", 1280))
        flux_model = body.get("flux_model", "dev")  # schnell | dev
        default_steps = 4 if flux_model == "schnell" else 20
        steps = int(body.get("steps", default_steps))
        quantize = int(body.get("quantize", 8))
        seed = body.get("seed", None)
        ref_b64 = body.get("ref_image_b64")
        ref_strength = float(body.get("ref_strength", 0.4))

        tmp_ref = None
        try:
            t = time.time()
            flux = get_flux(quantize, model_name=flux_model)
            logger.info(f"[generate/{flux_model}] {width}x{height} steps={steps} prompt={prompt[:60]}...")

            kwargs = dict(
                seed=seed or int(time.time()),
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
            )
            if ref_b64:
                tmp_ref = _save_tmp_png(ref_b64)
                kwargs["image_path"] = tmp_ref
                kwargs["image_strength"] = ref_strength

            image = flux.generate_image(**kwargs)
            buf = io.BytesIO()
            image.image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            elapsed = time.time() - t
            logger.info(f"완료: {elapsed:.1f}s")
            self._send_json(200, {"success": True, "image": f"data:image/png;base64,{b64}", "elapsed": elapsed})
        except Exception as e:
            logger.error(f"generate 실패: {e}", exc_info=True)
            self._send_json(500, {"success": False, "error": str(e)})
        finally:
            if tmp_ref:
                Path(tmp_ref).unlink(missing_ok=True)

    def _handle_kontext(self, body: dict):
        """Flux1 Kontext — 레퍼런스 이미지 기반 합성.

        body:
          prompt: str
          image_b64: str  — 콜라주 또는 단일 레퍼런스 이미지 (data URI)
          width, height, steps, guidance, quantize, seed
        """
        prompt = body.get("prompt", "")
        image_b64 = body.get("image_b64", "")
        width = int(body.get("width", 576))
        height = int(body.get("height", 1024))
        steps = int(body.get("steps", 8))
        guidance = float(body.get("guidance", 4.0))
        quantize = int(body.get("quantize", 8))
        seed = body.get("seed", None) or int(time.time())

        if not image_b64:
            self._send_json(400, {"success": False, "error": "image_b64 필수"})
            return

        tmp_img = None
        try:
            t = time.time()
            tmp_img = _save_tmp_png(image_b64)
            flux = get_flux_kontext(quantize)
            logger.info(f"[kontext] {width}x{height} steps={steps} guidance={guidance} prompt={prompt[:60]}...")

            image = flux.generate_image(
                seed=seed,
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance=guidance,
                image_path=tmp_img,
            )
            buf = io.BytesIO()
            image.image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            elapsed = time.time() - t
            logger.info(f"[kontext] 완료: {elapsed:.1f}s")
            self._send_json(200, {"success": True, "image": f"data:image/png;base64,{b64}", "elapsed": elapsed})
        except Exception as e:
            logger.error(f"kontext 실패: {e}", exc_info=True)
            self._send_json(500, {"success": False, "error": str(e)})
        finally:
            if tmp_img:
                Path(tmp_img).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()
    port = args.port

    if args.preload:
        get_flux()
        _last_request_time = time.time()
        _schedule_idle_shutdown()

    logger.info(f"mflux-server 시작 (port {port}, idle_timeout={IDLE_TIMEOUT}s)")
    server = HTTPServer(("127.0.0.1", port), Handler)
    logger.info(f"Ready: http://127.0.0.1:{port}")
    server.serve_forever()
