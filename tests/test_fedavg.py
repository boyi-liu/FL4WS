from contextlib import redirect_stderr
from importlib.util import find_spec
from io import StringIO
import json
import os
from pathlib import Path
import re
import select
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic
import unittest

import numpy as np

from main import main
from utils._training import load_training_config

if find_spec("torch") is not None:
    import torch

    from server.fedavg import FedAvgServer, aggregate
    from utils._protocol import receive_message, send_message
else:
    torch = None


class TrainingEntryPointTests(unittest.TestCase):
    def test_role_requires_client_id_only_for_clients(self):
        for options, message in (
            (["--role", "client"], "requires a nonnegative --client-id"),
            (["--role", "client", "--client-id", "-1"], "requires a nonnegative --client-id"),
            (["--role", "server", "--client-id", "0"], "only valid with --role client"),
        ):
            errors = StringIO()
            with self.subTest(options=options), redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
                main(["train", "--config", "unused.yaml", *options])
            self.assertEqual(2, raised.exception.code)
            self.assertIn(message, errors.getvalue())

    def test_config_resolves_relative_paths_and_rejects_invalid_settings(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "training.yaml"
            config_path.write_text("partition: data\nrounds: 3\n", encoding="utf-8")
            config = load_training_config(config_path)
            self.assertEqual(root / "data", config.partition)
            self.assertEqual(root / "global.pt", config.output)
            self.assertEqual(3, config.rounds)
            for setting, error in (("batch_size: 0", "positive integer"), ("batch_szie: 2", "unknown training settings")):
                with self.subTest(setting=setting), self.assertRaisesRegex(ValueError, error):
                    config_path.write_text(f"partition: data\n{setting}\n", encoding="utf-8")
                    load_training_config(config_path)


@unittest.skipIf(torch is None, "PyTorch model dependency is not installed")
class FedAvgTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "global.pt"
        self.config = self.root / "training.yaml"

    def _partition(self, counts):
        train = self.root / "train"
        train.mkdir()
        rng = np.random.default_rng(42)
        manifest = {"schema_version": 1, "dataset": "aril", "classes": ["up", "down"], "clients": []}
        offset = 0
        for client_id, count in enumerate(counts):
            manifest["clients"].append({"client_id": client_id, "train": list(range(offset, offset + count))})
            offset += count
            features = rng.standard_normal((count, 52, 192), dtype=np.float32)
            labels = np.full(count, client_id % 2, dtype=np.int64)
            np.savez(train / f"{client_id}.npz", data={"x": features, "y": labels})
        (self.root / "partition.json").write_text(json.dumps(manifest), encoding="utf-8")

    def _process(self, role, *arguments):
        entry = Path(__file__).resolve().parents[1] / "main.py"
        process = subprocess.Popen(
            [sys.executable, str(entry), "train", "--config", str(self.config), "--role", role, *arguments],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
        self.addCleanup(self._close_process, process)
        return process

    @staticmethod
    def _close_process(process):
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()

    def _server(self, fraction):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        self.config.write_text(json.dumps({
            "partition": ".", "port": port, "rounds": 2, "timeout": 30,
            "client_fraction": fraction, "batch_size": 2,
        }), encoding="utf-8")
        process = self._process("server")
        self.assertTrue(select.select([process.stdout], [], [], 20)[0], "Server did not start")
        line = process.stdout.readline()
        self.assertEqual(f"Listening on 127.0.0.1:{port}\n", line)
        return process, port

    def test_averages_by_training_samples_and_retains_integer_buffers(self):
        initial = {"weight": torch.tensor([0.0, 0.0]), "counter": torch.tensor(4)}
        updates = [
            ({"weight": torch.tensor([1.0, 5.0]), "counter": torch.tensor(6)}, 1),
            ({"weight": torch.tensor([3.0, 1.0]), "counter": torch.tensor(8)}, 3),
        ]
        result = aggregate(initial, updates)
        torch.testing.assert_close(result["weight"], torch.tensor([2.5, 2.0]))
        torch.testing.assert_close(result["counter"], torch.tensor(4))

    def test_independent_processes_train_only_when_selected(self):
        self._partition([2, 3, 5])
        server, _ = self._server(0.5)
        clients = [self._process("client", "--client-id", str(client_id)) for client_id in range(3)]
        log, _ = server.communicate(timeout=40)
        self.assertEqual(0, server.returncode, log)
        events = re.findall(r"^Round (\d+): (selected|aggregated) (.+)$", log, re.MULTILINE)
        self.assertEqual(
            [("1", "selected", "clients [0, 2]"), ("1", "aggregated", "2 updates"),
             ("2", "selected", "clients [0, 1]"), ("2", "aggregated", "2 updates")], events,
        )
        for client_id, (client, expected) in enumerate(zip(clients, (["1", "2"], ["2"], ["1"]))):
            output, _ = client.communicate(timeout=10)
            self.assertEqual(0, client.returncode, output)
            self.assertEqual(expected, re.findall(r"uploaded round (\d+) update", output))
            self.assertIn(f"Client {client_id}: training complete", output)

        final = torch.load(self.output, weights_only=True)
        model = FedAvgServer(self.root).model
        initial = model.state_dict()
        self.assertTrue(any(not torch.equal(initial[name], final[name]) for name in initial))
        model.load_state_dict(final)

    def test_wrong_round_is_rejected_without_aggregation(self):
        self._partition([2, 3])
        server, port = self._server(1.0)
        spec = FedAvgServer(self.root)
        deadline = monotonic() + 10
        with socket.create_connection(("127.0.0.1", port), timeout=5) as first, \
                socket.create_connection(("127.0.0.1", port), timeout=5) as second:
            for client_id, connection in enumerate((first, second)):
                send_message(connection, {
                    "type": "register", "client_id": client_id,
                    "partition": spec.partition.fingerprint, "schema": spec.codec.fingerprint,
                }, deadline=deadline)
            for connection, update_round in ((first, 1), (second, 2)):
                self.assertEqual({"type": "train", "round": 1}, receive_message(connection, deadline=deadline))
                state = spec.codec.receive(connection, deadline=deadline)
                send_message(connection, {"type": "update", "round": update_round},
                             spec.codec.encode(state), deadline=deadline)
            output, _ = server.communicate(timeout=10)
        self.assertNotEqual(0, server.returncode, output)
        self.assertIn("Expected an update for round 1; round not aggregated", output)
        self.assertNotIn("Round 2:", output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
