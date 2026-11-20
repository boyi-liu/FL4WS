"""Read the shared metadata for a federated training run."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Partition:
    root: Path
    dataset: str
    num_classes: int
    train_sizes: dict[int, int]
    test_sizes: dict[int, int]
    fingerprint: str


def load_partition(root):
    root = Path(root).expanduser()
    raw = (root / "partition.json").read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Expected an FL4WS partition.json with schema_version 1")
    dataset, classes, clients = (manifest.get(key) for key in ("dataset", "classes", "clients"))
    if not isinstance(dataset, str) or not isinstance(classes, list) or not classes:
        raise ValueError("Partition must specify a dataset and its global class list")
    if not isinstance(clients, list) or not clients:
        raise ValueError("Partition must contain at least one client")

    train_sizes = {}
    test_sizes = {}
    for client in clients:
        if not isinstance(client, dict):
            raise ValueError("Expected a client record in the partition manifest")
        client_id, train = client.get("client_id"), client.get("train")
        if type(client_id) is not int or client_id < 0 or client_id in train_sizes:
            raise ValueError(f"Invalid or duplicate client ID: {client_id!r}")
        if not isinstance(train, list) or not train:
            raise ValueError(f"Client {client_id} has no training samples")
        test = client.get("test", [])
        if not isinstance(test, list):
            raise ValueError(f"Client {client_id} has an invalid test assignment")
        train_sizes[client_id] = len(train)
        test_sizes[client_id] = len(test)
    return Partition(root, dataset, len(classes), train_sizes, test_sizes, sha256(raw).hexdigest())


def iter_test_batches(partition, sample_shape, *, batch_size):
    """Stream batches from trusted test archives, one client shard at a time."""
    from torch import from_numpy

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    for client_id, count in sorted(partition.test_sizes.items()):
        if not count:
            continue
        path = partition.root / "test" / f"{client_id}.npz"
        with np.load(path, allow_pickle=True) as archive:
            data = archive["data"].item()
        features, targets = data["x"], data["y"]
        if not isinstance(features, np.ndarray) or features.dtype != np.float32 or features.shape != (count, *sample_shape):
            raise ValueError(f"{path}: test features do not match the manifest and model shape")
        if not isinstance(targets, np.ndarray) or targets.dtype != np.int64 or targets.shape != (count,):
            raise ValueError(f"{path}: test labels do not match the manifest")
        for start in range(0, count, batch_size):
            batch = slice(start, start + batch_size)
            yield from_numpy(features[batch]), from_numpy(targets[batch])
