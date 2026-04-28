"""mFlux HTTP 서버 — Flux1 모델 상주 + 아이들 타임아웃.

- 첫 요청 시 모델 로드, 이후 재사용
- 마지막 요청 후 3분 경과 시 자동 종료 (Metal GPU 메모리 OS 반환)
- 다음 요청 시 diffusionkit_service가 자동 재기동

포트: 18190
"""

import argparse
import base64
import io
import json
import logging
import os
import signal
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

_flux = None
_loaded_quantize = None
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
    global _flux, _loaded_quantize
    elapsed = time.time() - _last_request_time
    if elapsed < IDLE_TIMEOUT - 5:
        # 요청이 다시 들어왔으면 취소
        _schedule_idle_shutdown()
        return
    logger.info(f"아이들 {IDLE_TIMEOUT}초 경과 → 프로세스 종료 (Metal GPU 메모리 반환)")
    os.kill(os.getpid(), signal.SIGTERM)


def get_flux(quantize: int = 8):
    global _flux, _loaded_quantize
    with _lock:
        if _flux is None or _loaded_quantize != quantize:
            logger.info(f"Flux1 dev 모델 로딩 (quantize={quantize})...")
            t = time.time()
            from mflux.models.common.config import ModelConfig
            from mflux.models.flux.variants.txt2img.flux import Flux1
            _flux = Flux1(
                model_config=ModelConfig.from_name(model_name="dev"),
                quantize=quantize,
            )
            _loaded_quantize = quantize
            logger.info(f"Flux1 dev 로딩 완료: {time.time() - t:.1f}s")
    return _flux


def _shutdown():
    logger.info("프로세스 종료")
    os.kill(os.getpid(), signal.SIGTERM)


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
            self._send_json(200, {"status": "ok", "engine": "mflux-dev", "loaded": _flux is not None})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        global _last_request_time

        if self.path == "/shutdown":
            self._send_json(200, {"success": True})
            Timer(0.3, _shutdown).start()
            return

        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except Exception:
            self._send_json(400, {"success": False, "error": "invalid JSON"})
            return

        prompt = body.get("prompt", "")
        width = int(body.get("width", 720))
        height = int(body.get("height", 1280))
        steps = int(body.get("steps", 20))
        quantize = int(body.get("quantize", 8))
        seed = body.get("seed", None)

        _last_request_time = time.time()

        try:
            t = time.time()
            flux = get_flux(quantize)
            logger.info(f"[generate] 시작: {width}x{height}, steps={steps}, prompt={prompt[:60]}...")

            image = flux.generate_image(
                seed=seed or int(time.time()),
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
            )

            buf = io.BytesIO()
            image.image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            elapsed = time.time() - t
            logger.info(f"완료: {elapsed:.1f}s, {len(buf.getvalue()) // 1024}KB")

            self._send_json(200, {
                "success": True,
                "image": f"data:image/png;base64,{b64}",
                "elapsed": elapsed,
            })

        except Exception as e:
            logger.error(f"생성 실패: {e}", exc_info=True)
            self._send_json(500, {"success": False, "error": str(e)})

        finally:
            # 매 요청 완료 후 아이들 타이머 리셋
            _last_request_time = time.time()
            _schedule_idle_shutdown()


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
