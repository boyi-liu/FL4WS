"""Train a single FL4WS client shard on demand over TCP."""

from math import isfinite
import socket
from time import monotonic

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import build_model
from utils._partition import load_partition
from utils._protocol import StateCodec, receive_message, send_message


class FedAvgClient:
    def __init__(self, partition, client_id, *, local_epochs=1, batch_size=32, lr=0.01, seed=42):
        if type(local_epochs) is not int or local_epochs < 1:
            raise ValueError("local_epochs must be a positive integer")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        self.partition = load_partition(partition)
        if type(client_id) is not int or client_id not in self.partition.train_sizes:
            raise ValueError(f"Client {client_id!r} is not in the partition")
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.lr = lr
        self.model = build_model(self.partition.dataset, num_classes=self.partition.num_classes)
        self.codec = StateCodec(self.model.state_dict())
        self.loader = DataLoader(
            self._load_shard(), batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed + client_id),
        )

    def _load_shard(self):
        path = self.partition.root / "train" / f"{self.client_id}.npz"
        with np.load(path, allow_pickle=True) as archive:
            data = archive["data"].item()
        features, labels = data["x"], data["y"]
        size = self.partition.train_sizes[self.client_id]
        if features.dtype != np.float32 or features.shape != (size, *self.model.sample_shape):
            raise ValueError(f"{path}: training features do not match the manifest and model shape")
        if labels.dtype != np.int64 or labels.shape != (size,) or not ((0 <= labels) & (labels < self.partition.num_classes)).all():
            raise ValueError(f"{path}: invalid training labels")
        return TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))

    def train(self, global_state):
        """Load this round's global weights and run local SGD from scratch."""
        self.model.load_state_dict(global_state)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        for _ in range(self.local_epochs):
            for features, labels in self.loader:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(self.model(features), labels)
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Client {self.client_id} produced a non-finite training loss")
                loss.backward()
                optimizer.step()
        return self.model.state_dict()

    def run(self, host, port, *, timeout=120.0):
        """Register and stay idle between rounds in which this client is selected."""
        if not isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        last_round = 0
        with socket.create_connection((host, port), timeout=timeout) as connection:
            send_message(connection, {
                "type": "register", "client_id": self.client_id,
                "partition": self.partition.fingerprint, "schema": self.codec.fingerprint,
            }, deadline=monotonic() + timeout)
            while True:
                message = receive_message(connection)
                if message.get("type") == "done":
                    print(f"Client {self.client_id}: training complete", flush=True)
                    return
                round_id = message.get("round")
                if message.get("type") != "train" or type(round_id) is not int or round_id <= last_round:
                    raise ValueError(f"Unexpected training message: {message!r}")
                global_state = self.codec.receive(connection, deadline=monotonic() + timeout)
                local_state = self.train(global_state)
                send_message(connection, {"type": "update", "round": round_id},
                             self.codec.encode(local_state), deadline=monotonic() + timeout)
                print(f"Client {self.client_id}: uploaded round {round_id} update", flush=True)
                last_round = round_id
