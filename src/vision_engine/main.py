"""CLI entry point for the headless vision engine."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/eyemouse_mpl"
os.environ["MPLBACKEND"] = "Agg"

import argparse
import logging
import sys

try:
    from .engine import VisionEngine
except (ImportError, SystemError, ValueError):
    try:
        from src.vision_engine.engine import VisionEngine
    except (ImportError, ValueError):
        from vision_engine.engine import VisionEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="EyeMouse Headless Vision Engine")
    parser.add_argument(
        "--socket",
        required=True,
        help="Unix domain socket path for host control communication",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging output",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    engine = VisionEngine(args.socket)
    try:
        engine.start()
        return 0
    except Exception as exc:
        logging.exception("Fatal error in vision engine: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
