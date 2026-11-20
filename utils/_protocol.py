"""Length-prefixed JSON headers and model tensors encoded as raw bytes."""

from hashlib import sha256
import json
from math import prod
from struct import Struct
from time import monotonic

import numpy as np
import torch

_HEADER_LENGTH = Struct("!I")
_MAX_HEADER_SIZE = 4096


def time_left(deadline):
    if deadline is None:
        return None
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("federated communication deadline expired")
    return remaining


def _receive_exact(connection, size, deadline):
    data = bytearray(size)
    remaining = memoryview(data)
    while remaining:
        connection.settimeout(time_left(deadline))
        count = connection.recv_into(remaining)
        if not count:
            raise ConnectionError("peer closed the connection")
        remaining = remaining[count:]
    return data


def send_message(connection, header, payload=b"", *, deadline=None):
    data = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(data) > _MAX_HEADER_SIZE:
        raise ValueError("Message header is too large")
    connection.settimeout(time_left(deadline))
    connection.sendall(_HEADER_LENGTH.pack(len(data)) + data)
    if payload:
        connection.settimeout(time_left(deadline))
        connection.sendall(payload)


def receive_message(connection, *, deadline=None):
    size, = _HEADER_LENGTH.unpack(_receive_exact(connection, _HEADER_LENGTH.size, deadline))
    if not 0 < size <= _MAX_HEADER_SIZE:
        raise ValueError("Invalid message header size")
    header = json.loads(_receive_exact(connection, size, deadline))
    if not isinstance(header, dict):
        raise ValueError("Expected a JSON object as the message header")
    return header


class StateCodec:
    """Use a common state_dict schema; no network data is unpickled."""

    def __init__(self, template):
        self.layout = {}
        self.size = 0
        for name, tensor in sorted(template.items()):
            dtype = tensor.detach().cpu().numpy().dtype.newbyteorder("<")
            shape = tuple(tensor.shape)
            self.layout[name] = (shape, dtype, self.size)
            self.size += tensor.numel() * dtype.itemsize
        schema = [(name, shape, dtype.str) for name, (shape, dtype, _) in self.layout.items()]
        self.fingerprint = sha256(json.dumps(schema).encode("utf-8")).hexdigest()

    def encode(self, state):
        if state.keys() != self.layout.keys():
            raise ValueError("Model state keys do not match the agreed schema")
        blocks = []
        for name, (shape, dtype, _) in self.layout.items():
            values = state[name].detach().cpu().numpy()
            if values.shape != shape or values.dtype.newbyteorder("<") != dtype:
                raise ValueError(f"Model tensor {name!r} does not match the agreed schema")
            blocks.append(values.astype(dtype, copy=False).tobytes())
        return b"".join(blocks)

    def receive(self, connection, *, deadline=None):
        payload = _receive_exact(connection, self.size, deadline)
        state = {}
        for name, (shape, dtype, offset) in self.layout.items():
            values = np.frombuffer(payload, dtype=dtype, count=prod(shape), offset=offset).reshape(shape)
            if not np.isfinite(values).all():
                raise ValueError(f"Model tensor {name!r} contains non-finite values")
            state[name] = torch.from_numpy(values.astype(dtype.newbyteorder("="), copy=True))
        return state
