"""Paths, telemetry serialization, and canonical numerical hashes."""
from pathlib import Path
import hashlib
import json
import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def load_params():
    return json.loads((RESULTS / "salted_params.json").read_text())


def write_json(path_or_name, data):
    path = Path(path_or_name)
    if not path.is_absolute():
        path = RESULTS / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest_array(array, sid):
    """SHA256(sid UTF-8 + rounded-to-6-decimal little-endian float64 C bytes)."""
    import numpy as np
    canonical = np.asarray(np.round(array, decimals=6), dtype="<f8", order="C")
    return hashlib.sha256(str(sid).encode("utf-8") + canonical.tobytes(order="C")).hexdigest()
