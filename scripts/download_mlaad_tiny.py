"""Download the public German MLAAD-tiny evaluation slice into ignored local data storage."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_yaml

logger = logging.getLogger("download_mlaad_tiny")


def download_mlaad_tiny(
    root: str | Path,
    *,
    repo_id: str,
    revision: str,
    language: str = "de",
) -> Path:
    """Fetch an explicitly selected language slice from the pinned public dataset revision.

    Generated samples are distributed under CC BY-NC 4.0. The bona-fide files
    retain their M-AILABS license in ``original/LICENSE``. The download stays in
    local ignored data storage and is not included in source control.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional downloader with `python -m pip install -r requirements-data.txt`."
        ) from exc

    language = language.lower()
    if language not in {"en", "de", "all"}:
        raise ValueError("language must be one of: en, de, all")
    root_path = Path(root).expanduser()
    if not root_path.is_absolute():
        root_path = PROJECT_ROOT / root_path

    languages = ("en", "de") if language == "all" else (language,)
    allow_patterns = [
        *(f"fake/{lang}/**" for lang in languages),
        *(f"original/{lang}/**" for lang in languages),
        "README.md",
        "LICENSE",
        "NOTICE_NO_COMMERCIAL_USE.txt",
        "original/LICENSE",
    ]
    root_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(root_path),
        allow_patterns=allow_patterns,
    )

    required = [root_path / "LICENSE", root_path / "original" / "LICENSE"]
    missing = [str(path) for path in required if not path.is_file()]
    missing.extend(
        str(root_path / "fake" / lang)
        for lang in languages
        if not (root_path / "fake" / lang).is_dir()
    )
    if missing:
        raise RuntimeError(
            "The download completed without expected dataset/license files: " + ", ".join(missing)
        )
    logger.info("MLAAD-tiny %s slice is available at %s (revision %s)", language, root_path, revision)
    return root_path


def parse_args() -> argparse.Namespace:
    """Parse config and optional local download overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--root", default=None, help="Local destination; defaults to configured MLAAD_TINY_ROOT.")
    parser.add_argument("--language", choices=("de", "en", "all"), default=None)
    return parser.parse_args()


def main() -> int:
    """Download the configured public subset without accepting gated dataset terms."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_yaml(args.config)
    spec = config.get("external_datasets", {}).get("mlaad_tiny", {})
    if not spec:
        raise ValueError(f"No external_datasets.mlaad_tiny section in {args.config}")
    root = args.root or spec.get("root")
    if not root:
        raise ValueError("Configure external_datasets.mlaad_tiny.root or pass --root")
    download_mlaad_tiny(
        root,
        repo_id=str(spec["repo_id"]),
        revision=str(spec["revision"]),
        language=args.language or str(spec.get("language", "de")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
