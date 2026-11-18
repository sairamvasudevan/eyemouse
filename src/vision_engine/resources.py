"""Model and asset resource management for development and frozen packaging."""

import os
import sys
from pathlib import Path


MODEL_FILENAME = "face_landmarker.task"
GEOMETRY_FILENAME = "geometry_pipeline_metadata_landmarks.binarypb"


class ResourceNotFoundError(FileNotFoundError):
    """Raised when an offline asset or model cannot be located on disk."""
    pass


def get_bundle_dir() -> Path:
    """Return base directory for bundled assets in development or frozen runtime."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    # Project root is 2 levels up from src/vision_engine/
    return Path(__file__).resolve().parent.parent.parent


def get_model_path(filename: str = MODEL_FILENAME) -> Path:
    """
    Resolve the absolute path to a model asset.
    
    Lookup precedence:
    1. PyInstaller frozen extraction bundle (_MEIPASS/assets/models/<filename>)
    2. Environment variable EYEMOUSE_MODEL_DIR
    3. Workspace repository assets/models/<filename>
    4. Package-local src/vision_engine/models/<filename>
    """
    env_dir = os.environ.get("EYEMOUSE_MODEL_DIR")
    candidates = []

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundle_dir = Path(sys._MEIPASS)
        candidates.append(bundle_dir / "assets" / "models" / filename)
        candidates.append(bundle_dir / filename)

    if env_dir:
        candidates.append(Path(env_dir) / filename)

    root_dir = get_bundle_dir()
    candidates.append(root_dir / "assets" / "models" / filename)
    candidates.append(Path(__file__).resolve().parent / "models" / filename)

    for path in candidates:
        if path.is_file():
            return path

    searched = "\n  - ".join(str(p) for p in candidates)
    raise ResourceNotFoundError(
        f"Model file '{filename}' was not found. Searched locations:\n  - {searched}\n"
        f"Please verify that the model has been placed in assets/models/."
    )


def get_face_landmarker_path() -> Path:
    """Get verified path to face_landmarker.task."""
    return get_model_path(MODEL_FILENAME)


def get_geometry_metadata_path() -> Path:
    """Get verified path to geometry_pipeline_metadata_landmarks.binarypb."""
    return get_model_path(GEOMETRY_FILENAME)

