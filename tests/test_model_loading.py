"""Automated tests for model loading and resource resolution."""

import pytest
from pathlib import Path
from src.vision_engine.resources import (
    get_face_landmarker_path,
    get_geometry_metadata_path,
    get_model_path,
    ResourceNotFoundError,
)


def test_face_landmarker_model_exists():
    path = get_face_landmarker_path()
    assert path.is_file(), f"Face landmarker model missing at {path}"
    assert path.stat().st_size > 1_000_000, "Model file size unexpectedly small"


def test_geometry_metadata_file_exists():
    path = get_geometry_metadata_path()
    assert path.is_file(), f"Geometry metadata missing at {path}"
    assert path.stat().st_size > 5_000, "Geometry metadata size unexpectedly small"


def test_missing_model_raises_explicit_error():
    with pytest.raises(ResourceNotFoundError) as exc_info:
        get_model_path("non_existent_model_file.task")
    assert "non_existent_model_file.task" in str(exc_info.value)

