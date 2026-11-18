"""User-provided MANO/SMPL-X models; importing this package never downloads files."""

import os
import subprocess
from pathlib import Path
from typing import Optional, Union


def get_body_models_dir() -> Path:
    """Return the local model root, optionally overridden by `ROBOKIT_BODY_MODELS`.

    Example:
        >>> isinstance(get_body_models_dir(), Path)
        True
    """
    return Path(os.environ.get("ROBOKIT_BODY_MODELS") or Path.home() / ".robokit" / "body_models").expanduser()


def get_model_path(model: str, variant: str, path: Optional[Union[str, Path]] = None) -> Path:
    """Find a local NPZ or convert an official provider's PKL once; imports perform no lookup or conversion.

    Path priority:
    1. Explicit path.
    2. `ROBOKIT_BODY_MODELS`.
    3. ~/.robokit/body_models.

    Conversion steps:
    1. Run the converter, using an isolated uv environment if SciPy is unavailable.
    2. Cache the NPZ beside the original PKL, leaving the PKL unchanged.

    Example:
        >>> # xdoctest: +REQUIRES(env:ROBOKIT_BODY_MODELS)
        >>> get_model_path("mano", "right").suffix
        '.npz'
    """
    variants = {"mano": ("left", "right"), "smplx": ("neutral", "male", "female")}
    if model not in variants or variant not in variants[model]:
        raise ValueError(f"Unknown body model/variant: {model}/{variant}")
    source = (
        Path(path).expanduser()
        if path is not None
        else get_body_models_dir() / model / f"{model.upper()}_{variant.upper()}.npz"
    )
    if source.suffix not in (".npz", ".pkl"):
        raise ValueError(f"Expected an official .npz or .pkl model: {source}")
    target, pkl = source.with_suffix(".npz"), source.with_suffix(".pkl")
    if target.is_file():
        return target
    if not pkl.is_file():
        url = "https://mano.is.tue.mpg.de/" if model == "mano" else "https://smpl-x.is.tue.mpg.de/"
        raise FileNotFoundError(
            f"Missing {model.upper()} {variant} model: {target}\n"
            f"1. Register, accept the license, and download the model from {url}\n"
            f"2. Extract {pkl.name} (or {target.name}) into {target.parent}.\n"
            "3. For another location, set ROBOKIT_BODY_MODELS to the folder containing mano/ and smplx/, "
            "or pass an explicit model path.\n"
            "Official PKLs are converted to NPZ automatically on first use. RoboKit never downloads these models."
        )
    from robokit.assets.body_models._convert import convert_model

    try:
        return convert_model(pkl, model, variant)
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] != "scipy":
            raise
    command = [
        "uv",
        "run",
        str(Path(__file__).with_name("_convert.py")),
        str(pkl),
        "--model",
        model,
        "--variant",
        variant,
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "PKL conversion requires uv when SciPy is absent. Install uv or supply an NPZ model."
        ) from error
    return target
