# /// script
# requires-python = ">=3.8"
# dependencies = ["numpy", "scipy", "filelock"]
# ///
"""Convert official local PKLs with isolated script dependencies; no RoboKit install needed."""

import argparse
import pickle
import tempfile
from pathlib import Path

import numpy as np
from filelock import FileLock


class _ChumpyArray:
    """Read legacy Chumpy leaf/selection arrays without importing Chumpy."""

    x: np.ndarray
    a: "_ChumpyArray"
    idxs: np.ndarray

    @property
    def r(self) -> np.ndarray:
        if hasattr(self, "x"):
            return self.x
        selected = self.a.r.reshape(-1)[self.idxs]
        return selected.reshape(getattr(self, "preferred_shape", selected.shape))


class _ModelUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in (("chumpy.ch", "Ch"), ("chumpy.reordering", "Select")):
            return _ChumpyArray
        allowed = (
            module
            in ("numpy", "numpy.core.multiarray", "numpy._core.multiarray", "numpy.core.numeric", "numpy._core.numeric")
            and name in ("ndarray", "dtype", "_reconstruct", "scalar", "_frombuffer")
            or module in ("scipy.sparse.csc", "scipy.sparse._csc", "scipy.sparse.csr", "scipy.sparse._csr")
            and name in ("csc_matrix", "csr_matrix")
            or module in ("builtins", "__builtin__")
            and name in ("set", "frozenset", "object")
            or module in ("copyreg", "copy_reg")
            and name == "_reconstructor"
            or module == "_codecs"
            and name == "encode"
        )
        if not allowed:
            raise pickle.UnpicklingError(
                f"Unsupported model pickle global: {module}.{name}; use an official model file"
            )
        if module.startswith("scipy.sparse."):
            module = "scipy.sparse"
        module = module.replace("numpy._core", "numpy.core")
        if int(np.__version__.split(".")[0]) >= 2:
            module = module.replace("numpy.core", "numpy._core")
        return super().find_class(module, name)


def convert_model(pkl: Path, model: str, variant: str) -> Path:
    """Convert a trusted PKL once, preserving its original and any existing NPZ.

    Example:
        >>> # xdoctest: +REQUIRES(env:ROBOKIT_BODY_MODELS)
        >>> import os
        >>> pkl = Path(os.environ["ROBOKIT_BODY_MODELS"]) / "mano/MANO_RIGHT.pkl"
        >>> convert_model(pkl, "mano", "right").suffix
        '.npz'
    """
    variants = {"mano": ("left", "right"), "smplx": ("neutral", "male", "female")}
    if model not in variants or variant not in variants[model] or pkl.suffix != ".pkl":
        raise ValueError(f"Expected an official PKL and valid model/variant, got {pkl}, {model}/{variant}")
    target = pkl.with_suffix(".npz")
    with FileLock(str(target) + ".lock"):
        if target.is_file():
            return target
        with pkl.open("rb") as stream:
            data = _ModelUnpickler(stream, encoding="latin1").load()
        fields = ["v_template", "shapedirs", "posedirs", "J_regressor", "kintree_table", "weights", "f"]
        fields += ["hands_components", "hands_mean"] if model == "mano" else ["hands_meanl", "hands_meanr"]
        if model == "smplx":
            fields += [key for key in ("hands_componentsl", "hands_componentsr") if key in data]
        arrays = {}
        for key in fields:
            value = data[key]
            if isinstance(value, _ChumpyArray):
                value = value.r
            if hasattr(value, "toarray"):
                value = value.toarray()
            arrays[key] = np.asarray(value)
            if arrays[key].dtype.hasobject:
                raise ValueError(f"Non-array field {key!r} in {pkl}; use an official model file")
        if model == "mano":
            side = pkl.stem.rsplit("_", 1)[-1].lower()
            arrays["side"] = np.asarray(data.get("side", side if side in variants[model] else variant))
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                np.savez(stream, **arrays)
                stream.close()
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pkl", type=Path)
    parser.add_argument("--model", required=True, choices=("mano", "smplx"))
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    print(convert_model(args.pkl.expanduser(), args.model, args.variant))
