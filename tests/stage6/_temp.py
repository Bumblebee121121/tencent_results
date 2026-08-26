from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def workspace_tempdir():
    root = Path.cwd() / "artifacts" / "stage6_test_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)

