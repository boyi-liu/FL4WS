"""Read the shared metadata for a federated training run."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path


@dataclass(frozen=True)
class Partition:
    root: Path
    dataset: str
    num_classes: int
    train_sizes: dict[int, int]
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
    for client in clients:
        if not isinstance(client, dict):
            raise ValueError("Expected a client record in the partition manifest")
        client_id, train = client.get("client_id"), client.get("train")
        if type(client_id) is not int or client_id < 0 or client_id in train_sizes:
            raise ValueError(f"Invalid or duplicate client ID: {client_id!r}")
        if not isinstance(train, list) or not train:
            raise ValueError(f"Client {client_id} has no training samples")
        train_sizes[client_id] = len(train)
    return Partition(root, dataset, len(classes), train_sizes, sha256(raw).hexdigest())
