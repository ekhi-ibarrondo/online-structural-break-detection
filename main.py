"""CrunchDAO entry point: ``train`` (nothing to train) and ``infer`` (one score per online point).

The model is pretrained. Weights are read from the runner's model directory if it holds
``committee.json``, else from ``resources/`` next to this file, else from
``$BREAKDETECT_WEIGHTS`` or ``weights/``.
"""

import os

# Single-threaded BLAS: multi-threaded reductions sum in a run-dependent order.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "1"
# Numba codegen for the generic x86-64 CPU, no SVML, no threads: the same machine code on every worker.
os.environ["NUMBA_CPU_NAME"] = "generic"
os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import breakdetect  # noqa: E402
from breakdetect.weights import default_weights_dir  # noqa: E402

# Fork workers over contiguous batches of series; each holds about 1.9 GiB.
INFER_PARALLELISM = 6

_MODEL = None


def _weights_dir(model_directory_path):
    for candidate in (model_directory_path, os.path.join(_HERE, "resources"), default_weights_dir()):
        if candidate and os.path.isfile(os.path.join(candidate, "committee.json")):
            return candidate
    raise FileNotFoundError("model weights not found (committee.json)")


def _model(model_directory_path=None):
    global _MODEL
    if _MODEL is None:
        _MODEL = breakdetect.load_model(_weights_dir(model_directory_path))
    return _MODEL


try:
    _model()  # load and compile before the runner forks its workers
except FileNotFoundError:
    pass


def train(model_directory_path):
    """Nothing to train: the committee ships pretrained."""
    return None


def infer(datasets, model_directory_path):
    """Yield once when ready, then one score in [0, 1] per online point, series after series."""
    model = _model(model_directory_path if isinstance(model_directory_path, str) else None)
    yield
    for item in datasets:
        try:
            x_historical, x_online = item
            yield from model.score_series(x_historical, x_online)
        except MemoryError:
            raise
        except Exception:
            continue
