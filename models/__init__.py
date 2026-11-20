"""Dataset-aware PyTorch classification baselines for FL4WS."""

from numbers import Integral

__all__ = ["build_model"]

_MODEL_SPECS = {
    "ut-har": ("cnn2d", (1, 250, 90), 7),
    "ntu-fi-har": ("cnn2d", (3, 114, 500), 6),
    "mm-fi": ("cnn-gru", (32, 3, 114, 10), 27),
    "wiar": ("cnn2d", (3, 30, 128), 16),
    "aril": ("cnn1d", (52, 192), 6),
    "widar3": ("cnn-gru", (22, 20, 20), 22),
}


def build_model(dataset, *, num_classes=None):
    """Build the baseline for a dataset ID, with an optional global class count."""
    if dataset not in _MODEL_SPECS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from {', '.join(_MODEL_SPECS)}.")
    architecture, sample_shape, default_classes = _MODEL_SPECS[dataset]
    if num_classes is None:
        num_classes = default_classes
    if isinstance(num_classes, bool) or not isinstance(num_classes, Integral) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")

    try:
        from .cnn1d import CSIConv1d
        from .cnn2d import CSIConv2d
        from .cnn2d_gru import CSISequenceClassifier
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        raise ImportError("PyTorch is required for models. From the repo, run `pip install -e '.[models]'`.") from error

    classifiers = {"cnn1d": CSIConv1d, "cnn2d": CSIConv2d, "cnn-gru": CSISequenceClassifier}
    return classifiers[architecture](sample_shape, int(num_classes))
