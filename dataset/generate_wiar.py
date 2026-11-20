"""Load the public WiAR participant archives as Intel 5300 CSI amplitudes."""

from hashlib import sha256
import re

import numpy as np

if __package__:
    from .utils import SampleMetadata, SensingDataset, run_generator
else:
    from utils import SampleMetadata, SensingDataset, run_generator


ACTIVITIES = (
    "horizontal arm wave", "high arm wave", "two hands wave", "high throw",
    "draw x", "draw tick", "toss paper", "forward kick", "side kick", "bend",
    "hand clap", "walk", "phone call", "drink water", "sit down", "squat",
)
ANTENNAS = 3
SUBCARRIERS = 30
PAYLOAD_BYTES = 192
# Each subcarrier has a three-bit prefix followed by signed 8-bit I/Q pairs.
BIT_POSITIONS = (
    np.arange(SUBCARRIERS)[:, None] * (3 + 16 * ANTENNAS)
    + 3 + 16 * np.arange(ANTENNAS)
)
BYTE_INDICES, BIT_SHIFTS = np.divmod(BIT_POSITIONS, 8)
BIT_SHIFTS = BIT_SHIFTS.astype(np.uint32)


class WiARDataset(SensingDataset):
    name = "wiar"
    packet_count = 128
    classes = ACTIVITIES

    def __init__(self, root, normalization):
        super().__init__(root, normalization)
        paths, metadata = [], []
        recordings = {}
        for path in sorted(self.root.glob("volunteer_*_all_data/csi_a*_*.dat")):
            subject = re.fullmatch(r"volunteer_([a-z0-9]+)_all_data", path.parent.name)
            filename = re.fullmatch(r"csi_a([0-9]+)_([0-9]+)\.dat", path.name)
            if subject is None or filename is None:
                raise ValueError(f"{path}: invalid WiAR participant directory or recording name")
            activity = int(filename[1]) - 1
            if not 0 <= activity < len(self.classes):
                raise ValueError(f"{path}: invalid WiAR activity number")
            subject_id = subject[1].upper()
            fingerprint = sha256(path.read_bytes()).digest()
            identity = (subject_id, activity)
            if fingerprint in recordings:
                if recordings[fingerprint] != identity:
                    raise ValueError(f"{path}: identical recording has conflicting participant or activity labels")
                continue
            recordings[fingerprint] = identity
            paths.append(path)
            metadata.append(SampleMetadata(
                path.relative_to(self.root).as_posix(), activity, subject_id=subject_id,
            ))
        if not paths:
            raise ValueError(f"{self.root}: expected volunteer_*_all_data/csi_a<activity>_<repeat>.dat")
        self._paths = tuple(paths)
        self.metadata = tuple(metadata)

    @property
    def preprocessing(self):
        return {
            **super().preprocessing,
            "packets_per_recording": self.packet_count,
            "deduplicate_exact_recordings": True,
        }

    def _read(self, index):
        path = self._paths[index]
        packets = _beamforming_packets(path.read_bytes())
        if not packets:
            raise ValueError(f"{path}: no complete 1-TX, 3-RX beamforming records")
        selection = np.linspace(0, len(packets) - 1, self.packet_count, dtype=int)
        payloads, permutations = zip(*(packets[i] for i in selection))
        wire = np.frombuffer(b"".join(payloads), dtype=np.uint8)
        wire = wire.reshape(self.packet_count, PAYLOAD_BYTES).astype(np.uint32)
        words = wire[:, BYTE_INDICES] | (wire[:, BYTE_INDICES + 1] << 8) | (wire[:, BYTE_INDICES + 2] << 16)
        words >>= BIT_SHIFTS
        real = words.astype(np.int8).astype(np.float32)
        imag = (words >> 8).astype(np.int8).astype(np.float32)
        amplitudes = np.hypot(real, imag)
        rx_order = np.argsort(permutations, axis=1)
        amplitudes = np.take_along_axis(amplitudes, rx_order[:, None, :], axis=2)
        return amplitudes.transpose(2, 1, 0)


def _beamforming_packets(raw):
    packets = []
    offset = 0
    while offset + 3 <= len(raw):
        length = int.from_bytes(raw[offset:offset + 2], "big")
        stop = offset + 2 + length
        if length == 0 or stop > len(raw):
            break  # Recorder shutdown can leave an incomplete final record.
        code = raw[offset + 2]
        record = raw[offset + 3:stop]
        offset = stop
        if code != 187 or len(record) < 20 + PAYLOAD_BYTES or record[8:10] != b"\x03\x01":
            continue
        if int.from_bytes(record[16:18], "little") != PAYLOAD_BYTES:
            continue
        antenna_sel = record[15]
        permutation = tuple((antenna_sel >> shift) & 3 for shift in (0, 2, 4))
        if set(permutation) != {0, 1, 2}:
            permutation = (0, 1, 2)
        packets.append((record[20:20 + PAYLOAD_BYTES], permutation))
    return packets


def load_dataset(root, *, normalization="none"):
    return WiARDataset(root, normalization)


if __name__ == "__main__":
    run_generator(load_dataset, dataset_name=WiARDataset.name)
