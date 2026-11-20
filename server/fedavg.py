"""Run synchronous FedAvg over persistent TCP client connections."""

from contextlib import ExitStack
import json
from math import ceil, isfinite
from pathlib import Path
import random
import socket
from time import monotonic

import torch

from models import build_model
from utils._partition import iter_test_batches, load_partition
from utils._protocol import StateCodec, receive_message, send_message, time_left
from utils.evaluation import evaluate


def aggregate(global_state, updates):
    """Average floating-point tensors by sample count and keep other server tensors."""
    updates = tuple(updates)
    if not updates or any(sample_count <= 0 for _, sample_count in updates):
        raise ValueError("FedAvg requires a positive sample count for each client update")
    total = sum(sample_count for _, sample_count in updates)
    result = {}
    for name, reference in global_state.items():
        if not reference.is_floating_point():
            result[name] = reference
            continue
        average = torch.zeros_like(reference)
        for state, sample_count in updates:
            average.add_(state[name], alpha=sample_count / total)
        result[name] = average
    return result


class FedAvgServer:
    def __init__(self, partition, *, rounds=5, client_fraction=1.0, seed=42):
        if type(rounds) is not int or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        if not 0 < client_fraction <= 1:
            raise ValueError("client_fraction must be in (0, 1]")
        self.partition = load_partition(partition)
        self.rounds = rounds
        self.client_count = ceil(client_fraction * len(self.partition.train_sizes))
        self.random = random.Random(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = build_model(self.partition.dataset, num_classes=self.partition.num_classes)
        self.codec = StateCodec(self.model.state_dict())
        self.metrics_by_round = {}

    def _register_clients(self, listener, stack, deadline):
        connections = {}
        while len(connections) < len(self.partition.train_sizes):
            listener.settimeout(time_left(deadline))
            connection, _ = listener.accept()
            stack.enter_context(connection)
            message = receive_message(connection, deadline=deadline)
            client_id = message.get("client_id")
            if message.get("type") != "register" or type(client_id) is not int:
                raise ValueError("Expected registration with an integer client ID")
            if client_id not in self.partition.train_sizes or client_id in connections:
                raise ValueError(f"Unknown or already connected client {client_id}")
            if message.get("partition") != self.partition.fingerprint or message.get("schema") != self.codec.fingerprint:
                raise ValueError(f"Client {client_id} uses a different partition or model schema")
            connections[client_id] = connection
            print(f"Registered client {client_id}", flush=True)
        return connections

    def _run_round(self, round_id, connections, timeout):
        selected = sorted(self.random.sample(sorted(connections), self.client_count))
        deadline = monotonic() + timeout
        print(f"Round {round_id}: selected clients {selected}", flush=True)
        state = self.model.state_dict()
        payload = self.codec.encode(state)
        for client_id in selected:
            try:
                send_message(connections[client_id], {"type": "train", "round": round_id}, payload, deadline=deadline)
            except OSError as error:
                raise RuntimeError(f"Round {round_id}, client {client_id}: {error}; round not aggregated") from error

        updates = []
        for client_id in selected:
            connection = connections[client_id]
            try:
                message = receive_message(connection, deadline=deadline)
                update_round = message.get("round")
                if message.get("type") != "update" or type(update_round) is not int or update_round != round_id:
                    raise ValueError(f"Expected an update for round {round_id}")
                update = self.codec.receive(connection, deadline=deadline)
            except (OSError, ValueError) as error:
                raise RuntimeError(f"Round {round_id}, client {client_id}: {error}; round not aggregated") from error
            updates.append((update, self.partition.train_sizes[client_id]))
        self.model.load_state_dict(aggregate(state, updates))
        print(f"Round {round_id}: aggregated {len(updates)} updates", flush=True)

    def evaluate(self, *, batch_size=32):
        """Evaluate the current global model on all held-out test shards."""
        batches = iter_test_batches(self.partition, self.model.sample_shape, batch_size=batch_size)
        return evaluate(self.model, batches, num_classes=self.partition.num_classes)

    def run(self, host, port, *, timeout=120.0, output=None, eval_every=0, eval_batch_size=32):
        """Wait for the manifest roster and train, returning the final global model."""
        if not isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if type(eval_every) is not int or eval_every < 0:
            raise ValueError("eval_every must be a nonnegative integer")
        if eval_every and not any(self.partition.test_sizes.values()):
            raise ValueError("Round evaluation requires test samples in partition.json")
        self.metrics_by_round.clear()
        with socket.create_server((host, port)) as listener, ExitStack() as stack:
            address = listener.getsockname()
            print(f"Listening on {address[0]}:{address[1]}", flush=True)
            connections = self._register_clients(listener, stack, monotonic() + timeout)
            for round_id in range(1, self.rounds + 1):
                self._run_round(round_id, connections, timeout)
                if eval_every and round_id % eval_every == 0:
                    metrics = self.evaluate(batch_size=eval_batch_size)
                    self.metrics_by_round[round_id] = metrics
                    print(f"Round {round_id}: evaluation {json.dumps(metrics)}", flush=True)
            if output is not None:
                output = Path(output).expanduser()
                output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), output)
                print(f"Saved global model to {output}", flush=True)
            for client_id, connection in connections.items():
                try:
                    send_message(connection, {"type": "done"}, deadline=monotonic() + timeout)
                except OSError as error:
                    print(f"Could not notify client {client_id} of completion: {error}", flush=True)
        return self.model
