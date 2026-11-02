"""Serve the local browser interface for one-clip AV detector inference."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.clip_service import ClipPredictionService, InferenceInputError
from src.utils.config_loader import load_yaml

logger = logging.getLogger("serve_web")


def parse_args() -> argparse.Namespace:
    """Read checkpoint and optional project-config overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Full AV checkpoint from reliability-fusion training.")
    parser.add_argument("--web-config", default="configs/web_app_config.yaml")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--eval-config", default="configs/eval_config.yaml")
    parser.add_argument("--manifest-dir", default="manifests/splits")
    parser.add_argument("--host", default=None, help="Defaults to the local-only host in web_app_config.yaml.")
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


class ReviewRequestHandler(BaseHTTPRequestHandler):
    """Serve one static page and local same-origin JSON inference endpoints."""

    service: ClipPredictionService
    page_path: Path

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        """Return a small JSON response with explicit content length."""
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Serve the application shell or a local readiness summary."""
        route = urlsplit(self.path).path
        if route == "/":
            body = self.page_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if route == "/api/status":
            self._send_json(200, self.service.status())
            return
        self._send_json(404, {"error": "Route not found"})

    def do_POST(self) -> None:
        """Receive one raw video body, score it, and remove its temporary file."""
        if urlsplit(self.path).path != "/api/predict":
            self._send_json(404, {"error": "Route not found"})
            return
        origin = self.headers.get("Origin")
        if origin:
            parsed_origin = urlsplit(origin)
            if parsed_origin.scheme != "http" or parsed_origin.netloc != self.headers.get("Host"):
                self._send_json(403, {"error": "Requests must come from this local app."})
                return
        content_type = self.headers.get("Content-Type", "").lower()
        if not (content_type.startswith("video/") or content_type == "application/octet-stream"):
            self._send_json(415, {"error": "Send one video file as the request body."})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length header"})
            return
        if content_length <= 0:
            self._send_json(400, {"error": "Choose a video file first."})
            return
        if content_length > self.service.max_upload_bytes:
            self._send_json(
                413,
                {"error": f"The upload limit is {self.service.max_upload_bytes // (1024 * 1024)} MB."},
            )
            return

        filename = unquote(self.headers.get("X-File-Name", "clip.mp4"))
        suffix = Path(filename).suffix.lower()
        if suffix not in self.service.allowed_extensions:
            self._send_json(415, {"error": "Use MP4, MOV, MKV, AVI, or WebM video."})
            return

        try:
            with tempfile.TemporaryDirectory(prefix="av-ui-upload-") as temp_dir:
                upload_path = Path(temp_dir) / f"upload{suffix}"
                remaining = content_length
                with upload_path.open("wb") as destination:
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise InferenceInputError("The upload ended before the full file arrived.")
                        destination.write(chunk)
                        remaining -= len(chunk)
                result = self.service.predict(upload_path)
            self._send_json(200, result)
        except InferenceInputError as exc:
            self._send_json(422, {"error": str(exc)})
        except FileNotFoundError as exc:
            logger.error("A local model or validation resource is unavailable: %s", exc)
            self._send_json(503, {"error": "A model or local validation resource is missing. Check the server log."})
        except Exception:
            logger.exception("Local clip inference failed")
            self._send_json(
                500,
                {"error": "Could not process this clip. Try a short H.264 MP4 with a visible face and clear audio."},
            )

    def log_message(self, format: str, *args: object) -> None:
        """Keep normal request logs concise and avoid echoing uploaded filenames."""
        logger.info("%s %s", self.command, urlsplit(self.path).path)


def main() -> int:
    """Start the local-only web app and validate its checkpoint before serving."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    web_config_path = Path(args.web_config)
    if not web_config_path.is_absolute():
        web_config_path = PROJECT_ROOT / web_config_path
    raw_config = load_yaml(web_config_path).get("server", {})
    host = args.host or str(raw_config["host"])
    port = int(args.port if args.port is not None else raw_config["port"])

    service = ClipPredictionService(
        args.checkpoint,
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        eval_config_path=args.eval_config,
        manifest_dir=args.manifest_dir,
        web_config_path=web_config_path,
    )
    service.load_model()

    handler = type(
        "ConfiguredReviewRequestHandler",
        (ReviewRequestHandler,),
        {"service": service, "page_path": PROJECT_ROOT / "web" / "index.html"},
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    logger.info("Local AV detector ready at http://%s:%d", host, port)
    logger.info("Checkpoint: %s; device: %s", service.checkpoint_path, service.device)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping local web app")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
