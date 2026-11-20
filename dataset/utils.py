"""Shared CSI loading, federated partitioning, and export utilities."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
from scipy.io import loadmat
import yaml


@dataclass(frozen=True)
class SampleMetadata:
    sample_id: str
    label: int
    subject_id: Optional[str] = None
    environment_id: Optional[str] = None
    source_split: Optional[str] = None


class SensingDataset:
    """Indexable dataset returning (float32 CSI, integer label)."""

    def __init__(self, root, normalization):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")
        if normalization not in ("none", "zscore"):
            raise ValueError("normalization must be 'none' or 'zscore'")
        self.normalization = normalization

    def __len__(self):
        return len(self.metadata)

    @property
    def preprocessing(self):
        return {"normalization": self.normalization}

    def __getitem__(self, index):
        meta = self.metadata[index]
        x = np.array(self._read(index), dtype=np.float32, copy=True, order="C")
        if not np.isfinite(x).all():
            raise ValueError(f"{meta.sample_id}: non-finite CSI values")
        if self.normalization == "zscore":
            scale = x.std(dtype=np.float64)
            centered = x.astype(np.float64) - x.mean(dtype=np.float64)
            x = (centered / (scale or 1.0)).astype(np.float32)
        return x, meta.label


def read_csi_amplitude(path):
    amplitude = loadmat(path, variable_names=["CSIamp"]).get("CSIamp")
    if amplitude is None or not amplitude.size or np.iscomplexobj(amplitude):
        raise ValueError(f"{path}: expected a nonempty real CSIamp array")
    return amplitude


GROUP_FIELDS = {
    "cross-subject": "subject_id",
    "cross-environment": "environment_id",
}
STRATEGIES = ("iid", "dirichlet", *GROUP_FIELDS)


@dataclass(frozen=True)
class PartitionConfig:
    strategy: str
    seed: int = 42
    num_clients: Optional[int] = None
    alpha: float = 0.5
    train_ratio: float = 0.8
    min_train_samples: int = 1

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy {self.strategy!r}. Choose from {', '.join(STRATEGIES)}.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.num_clients is not None and (type(self.num_clients) is not int or self.num_clients < 1):
            raise ValueError("num_clients must be a positive integer")
        if self.strategy not in GROUP_FIELDS and self.num_clients is None:
            raise ValueError(f"{self.strategy} requires num_clients")
        if type(self.min_train_samples) is not int or self.min_train_samples < 1:
            raise ValueError("min_train_samples must be a positive integer")
        if not isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")


@dataclass(frozen=True)
class ClientPartition:
    client_id: int
    train: tuple[int, ...]
    test: tuple[int, ...]
    group_id: Optional[str] = None


@dataclass(frozen=True)
class PartitionResult:
    dataset: SensingDataset
    config: PartitionConfig
    evaluation_protocol: str
    clients: tuple[ClientPartition, ...]


def supported_strategies(dataset: SensingDataset):
    """Return strategies whose required metadata is populated in this dataset."""
    natural = []
    for strategy, field in GROUP_FIELDS.items():
        groups = {getattr(meta, field) for meta in dataset.metadata}
        if None not in groups and len(groups) >= 2:
            natural.append(strategy)
    return ("iid", "dirichlet", *natural)


def partition_dataset(dataset: SensingDataset, config: PartitionConfig) -> PartitionResult:
    """Assign every sample exactly once, keeping evaluation independent of the strategy."""
    if config.strategy not in supported_strategies(dataset):
        field = GROUP_FIELDS[config.strategy]
        raise ValueError(f"{dataset.name} requires at least two complete {field} groups for {config.strategy}")

    train, test, protocol = _evaluation_pools(dataset, config)
    if config.strategy in GROUP_FIELDS:
        field = GROUP_FIELDS[config.strategy]
        train_groups = _group_indices(dataset, train, field)
        test_groups = _group_indices(dataset, test, field)
        groups = sorted(train_groups.keys() | test_groups.keys())
        if config.num_clients is not None and config.num_clients != len(groups):
            raise ValueError(f"{config.strategy} has exactly {len(groups)} metadata groups")
        clients = tuple(
            ClientPartition(client_id, tuple(train_groups[group]), tuple(test_groups[group]), group)
            for client_id, group in enumerate(groups)
        )
    else:
        count = config.num_clients
        if len(train) < count * config.min_train_samples or len(test) < count:
            raise ValueError("Too few samples for num_clients and min_train_samples in the evaluation pools")
        labels = np.array([meta.label for meta in dataset.metadata], dtype=np.int64)
        train_rng = np.random.default_rng([config.seed, 1])
        test_rng = np.random.default_rng([config.seed, 2])
        if config.strategy == "iid":
            train_clients = _iid(train, labels, count, train_rng)
        else:
            train_clients = _dirichlet(train, labels, config, train_rng)
        test_clients = _iid(test, labels, count, test_rng)
        clients = tuple(
            ClientPartition(client_id, tuple(train_indices), tuple(test_indices))
            for client_id, (train_indices, test_indices) in enumerate(zip(train_clients, test_clients))
        )

    for client in clients:
        if len(client.train) < config.min_train_samples or not client.test:
            raise ValueError(
                f"Client {client.client_id} (group {client.group_id!r}) has "
                f"{len(client.train)} train and {len(client.test)} test samples. "
                f"Each client needs at least {config.min_train_samples} train and one test sample."
            )
    return PartitionResult(dataset, config, protocol, clients)


def _evaluation_pools(dataset, config):
    if any(meta.source_split is not None for meta in dataset.metadata):
        train = [i for i, meta in enumerate(dataset.metadata) if meta.source_split == "train"]
        test = [i for i, meta in enumerate(dataset.metadata) if meta.source_split in ("val", "test")]
        if not train or not test or len(train) + len(test) != len(dataset):
            raise ValueError("Expected a complete source train and held-out split")
        return train, test, "source-train-vs-held-out"

    # Hold out whole recordings within each subject, independently of client assignment.
    subjects = _group_indices(dataset, range(len(dataset)), "subject_id")
    rng = np.random.default_rng([config.seed, 0])
    train, test = [], []
    for subject in sorted(subjects):
        rows = rng.permutation(subjects[subject]).tolist()
        cutoff = max(1, int(len(rows) * config.train_ratio))
        train.extend(rows[:cutoff])
        test.extend(rows[cutoff:])
    return sorted(train), sorted(test), "within-subject-recording-holdout"


def _group_indices(dataset, indices, field):
    groups = defaultdict(list)
    for index in indices:
        groups[getattr(dataset.metadata[index], field)].append(index)
    return groups


def _by_label(indices, labels, rng):
    indices = np.asarray(indices, dtype=np.int64)
    local_labels = labels[indices]
    for label in np.unique(local_labels):
        yield rng.permutation(indices[local_labels == label])


def _iid(indices, labels, count, rng):
    clients = [[] for _ in range(count)]
    order = rng.permutation(count)
    offset = 0
    for rows in _by_label(indices, labels, rng):
        for slot, client in enumerate(order):
            clients[client].extend(rows[(slot - offset) % count :: count].tolist())
        offset = (offset + len(rows)) % count
    return [rng.permutation(rows).tolist() for rows in clients]


def _dirichlet(indices, labels, config, rng):
    count = config.num_clients
    max_attempts = 256
    for _ in range(max_attempts):
        clients = [[] for _ in range(count)]
        for rows in _by_label(indices, labels, rng):
            proportions = _dirichlet_proportions(config.alpha, count, rng)
            sizes = rng.multinomial(len(rows), proportions)
            for client, assigned in zip(clients, np.split(rows, sizes.cumsum()[:-1])):
                client.extend(assigned.tolist())
        if min(map(len, clients)) >= config.min_train_samples:
            return [rng.permutation(rows).tolist() for rows in clients]
    raise ValueError(
        f"Dirichlet could not meet min_train_samples in {max_attempts} draws. "
        "Use a larger alpha, fewer clients, or a smaller minimum."
    )


def _dirichlet_proportions(alpha, count, rng):
    if alpha < 1:
        # A Gamma(alpha + 1) draw times U ** (1 / alpha) is Gamma(alpha).
        # Rescale its log before dividing to avoid underflow for small alpha.
        log_weights = alpha * np.log(rng.standard_gamma(alpha + 1, size=count))
        log_weights += np.log1p(-rng.random(count))
        with np.errstate(over="ignore"):
            weights = np.exp((log_weights - log_weights.max()) / alpha)
    else:
        weights = rng.standard_gamma(alpha, size=count)
        weights /= weights.max()
    return weights / weights.sum()


def write_partition(result: PartitionResult, output):
    """Publish a complete output directory after all requested writes succeed."""
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fl4ws-", dir=output.parent) as temporary:
        staging = Path(temporary) / "partition"
        staging.mkdir()
        _write_json(staging / "partition.json", _manifest(result))
        _write_json(staging / "statistics.json", _statistics(result))
        with (staging / "config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(
                {
                    "dataset": result.dataset.name,
                    **asdict(result.config),
                    "normalization": result.dataset.normalization,
                },
                stream,
                sort_keys=False,
            )
        save_data(staging, result)
        staging.rename(output)
    return output


def _write_json(path, payload):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _manifest(result):
    dataset = result.dataset
    return {
        "schema_version": 1,
        "numpy_version": np.__version__,
        "dataset": dataset.name,
        "classes": list(dataset.classes),
        "preprocessing": dataset.preprocessing,
        "partition_config": asdict(result.config),
        "evaluation_protocol": result.evaluation_protocol,
        "samples": [asdict(meta) for meta in dataset.metadata],
        "clients": [asdict(client) for client in result.clients],
    }


def _statistics(result):
    def summarize(indices):
        rows = [result.dataset.metadata[i] for i in indices]
        labels = Counter(row.label for row in rows)
        subjects = Counter(row.subject_id for row in rows if row.subject_id is not None)
        environments = Counter(
            row.environment_id for row in rows if row.environment_id is not None
        )
        return {
            "num_samples": len(rows),
            "label_counts": {i: labels[i] for i in range(len(result.dataset.classes))},
            "subject_counts": dict(sorted(subjects.items())),
            "environment_counts": dict(sorted(environments.items())),
        }

    train = [i for client in result.clients for i in client.train]
    test = [i for client in result.clients for i in client.test]
    return {
        "num_clients": len(result.clients),
        "train": summarize(train),
        "test": summarize(test),
        "clients": [
            {
                "client_id": client.client_id,
                "group_id": client.group_id,
                "train": summarize(client.train),
                "test": summarize(client.test),
            }
            for client in result.clients
        ],
    }


def save_data(directory: Path, result: PartitionResult):
    """Write a NumPy archive for each client's train and test samples."""
    for split in ("train", "test"):
        (directory / split).mkdir()
    dataset = result.dataset
    shape = None
    for client in result.clients:
        for split, indices in (("train", client.train), ("test", client.test)):
            for row, index in enumerate(indices):
                sample, label = dataset[index]
                if shape is None:
                    shape = sample.shape
                if sample.shape != shape:
                    raise ValueError(
                        f"{dataset.metadata[index].sample_id}: "
                        f"expected feature shape {shape}, got {sample.shape}"
                    )
                if row == 0:
                    x = np.empty((len(indices), *shape), dtype=np.float32)
                    y = np.empty(len(indices), dtype=np.int64)
                x[row], y[row] = sample, label
            np.savez_compressed(directory / split / f"{client.client_id}.npz", data={"x": x, "y": y})
            del x, y

def _generator_options(args):
    config_path = args.config.expanduser()
    with config_path.open(encoding="utf-8") as stream:
        options = yaml.safe_load(stream)
    if options is None:
        options = {}
    if not isinstance(options, dict) or any(not isinstance(key, str) for key in options):
        raise ValueError(f"{config_path}: expected a YAML mapping with string keys")
    overrides = {key: value for key, value in vars(args).items() if key != "config"}
    # The unified command and the dataset scripts can share one config file.
    unknown = options.keys() - overrides.keys() - {"dataset"}
    if unknown:
        raise ValueError(f"{config_path}: unknown settings: {', '.join(sorted(unknown))}")
    options.update({key: value for key, value in overrides.items() if value is not None})
    return options


def run_generator(load_dataset, argv=None, *, datasets=(), dataset_name=None):
    """Run a dataset script or the CLI for all registered datasets."""
    parser = argparse.ArgumentParser(description="Prepare federated WiFi CSI data partitions.")
    add_generator_arguments(parser, datasets=datasets)
    run_partition_command(load_dataset, parser, parser.parse_args(argv), dataset_name=dataset_name)


def add_generator_arguments(parser, *, datasets=()):
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="YAML settings file (default: dataset/config.yaml)",
    )
    if datasets:
        parser.add_argument("--dataset", choices=datasets)
    parser.add_argument("--root", help="Extracted dataset directory")
    parser.add_argument("--strategy", choices=STRATEGIES)
    parser.add_argument("--output", help="New directory for client data, configuration, and statistics")
    parser.add_argument("--num-clients", type=int, help="Required for IID and Dirichlet")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--alpha", type=float, help="Dirichlet concentration")
    parser.add_argument("--min-train-samples", type=int)
    parser.add_argument("--train-ratio", type=float, help="Recording holdout for datasets without a source split")
    parser.add_argument("--normalization", choices=("none", "zscore"))


def run_partition_command(load_dataset, parser, args, *, dataset_name=None):
    try:
        options = _generator_options(args)
        unified = hasattr(args, "dataset")
        required = ("dataset", "root", "output", "strategy") if unified else ("root", "output", "strategy")
        missing = [key for key in required if options.get(key) is None]
        if missing:
            raise ValueError(f"Set {', '.join(missing)} in the config file or on the command line")

        name = options.pop("dataset", None)
        if dataset_name is not None and name not in (None, dataset_name):
            raise ValueError(f"This generator uses {dataset_name}; the config specifies {name!r}")
        root = options.pop("root")
        output = options.pop("output")
        normalization = options.pop("normalization", "none")
        config = PartitionConfig(**options)
        loader_args = (name, root) if unified else (root,)
        dataset = load_dataset(*loader_args, normalization=normalization)
        result = partition_dataset(dataset, config)
        output = write_partition(result, output)
    except (ValueError, TypeError, OSError, NotImplementedError, yaml.YAMLError) as error:
        parser.error(str(error))
    print(f"Wrote {len(result.clients)} clients and {len(dataset)} sample assignments to {output}")
