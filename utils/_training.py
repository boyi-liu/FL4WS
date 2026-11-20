"""Shared YAML configuration for the FedAvg entry point."""

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    partition: Path
    output: Path
    host: str = "127.0.0.1"
    port: int = 8765
    rounds: int = 5
    client_fraction: float = 1.0
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.01
    seed: int = 42
    timeout: float = 120.0

    def __post_init__(self):
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must be a nonempty string")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("port must be an integer in [1, 65535]")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be a nonnegative 63-bit integer")
        for field in ("rounds", "local_epochs", "batch_size"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if type(self.client_fraction) not in (int, float) or not 0 < self.client_fraction <= 1:
            raise ValueError("client_fraction must be in (0, 1]")
        for field in ("lr", "timeout"):
            value = getattr(self, field)
            if type(value) not in (int, float) or not 0 < value < float("inf"):
                raise ValueError(f"{field} must be finite and positive")


def load_training_config(path):
    path = Path(path).expanduser().absolute()
    with path.open(encoding="utf-8") as stream:
        options = yaml.safe_load(stream)
    if not isinstance(options, dict) or any(not isinstance(key, str) for key in options):
        raise ValueError(f"{path}: expected a YAML mapping with string keys")
    unknown = options.keys() - {field.name for field in fields(TrainingConfig)}
    if unknown:
        raise ValueError(f"{path}: unknown training settings: {', '.join(sorted(unknown))}")
    options.setdefault("output", "global.pt")
    for key in ("partition", "output"):
        value = options.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: {key} must be a nonempty path")
        options[key] = path.parent / Path(value).expanduser()
    return TrainingConfig(**options)
