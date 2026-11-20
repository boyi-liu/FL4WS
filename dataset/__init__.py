"""CSI dataset generators and their shared federated partitioning API."""

from importlib import import_module

from .utils import (
    ClientPartition,
    PartitionConfig,
    PartitionResult,
    SampleMetadata,
    SensingDataset,
    partition_dataset,
    supported_strategies,
    write_partition,
)

DATASETS = ("ut-har", "ntu-fi-har", "mm-fi", "wiar", "aril", "widar3")

__all__ = [
    "ClientPartition",
    "PartitionConfig",
    "PartitionResult",
    "SampleMetadata",
    "SensingDataset",
    "load_dataset",
    "partition_dataset",
    "supported_strategies",
    "write_partition",
]


def load_dataset(name, root, *, normalization="none"):
    """Open an extracted dataset root with deterministic sample and label IDs."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {', '.join(DATASETS)}.")
    generator = import_module(f".generate_{name.replace('-', '_')}", package=__name__)
    return generator.load_dataset(root, normalization=normalization)
