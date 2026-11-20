"""Load MM-Fi WiFi CSI and generate its federated data partitions."""

import re

import numpy as np

if __package__:
    from .utils import SampleMetadata, SensingDataset, read_csi_amplitude, run_generator
else:
    from utils import SampleMetadata, SensingDataset, read_csi_amplitude, run_generator


class MMFiDataset(SensingDataset):
    name = "mm-fi"
    frame_count = 32

    @property
    def preprocessing(self):
        return {**super().preprocessing, "frames_per_recording": self.frame_count}

    def __init__(self, root, normalization):
        super().__init__(root, normalization)
        recordings = []
        for directory in sorted(self.root.glob("E[0-9][0-9]/S[0-9][0-9]/A[0-9][0-9]/wifi-csi")):
            frames = sorted(
                (
                    frame
                    for frame in directory.glob("frame*.mat")
                    if re.fullmatch(r"frame[0-9]+", frame.stem) and frame.stat().st_size > 0
                ),
                key=lambda frame: int(frame.stem[5:]),
            )
            if frames:
                recordings.append((directory.parent.relative_to(self.root).parts, frames))
        if not recordings:
            raise ValueError(f"{self.root}: no nonempty MAT recordings in E##/S##/A##/wifi-csi/")

        self.classes = tuple(sorted({action for (_, _, action), _ in recordings}))
        self.metadata = tuple(
            SampleMetadata(
                f"{environment}/{subject}/{action}",
                self.classes.index(action),
                subject_id=subject,
                environment_id=environment,
            )
            for (environment, subject, action), _ in recordings
        )
        self._frames = tuple(
            tuple(frames[i] for i in np.linspace(0, len(frames) - 1, self.frame_count, dtype=int))
            for _, frames in recordings
        )

    def _read(self, index):
        frames = {}
        for path in dict.fromkeys(self._frames[index]):
            x = read_csi_amplitude(path).astype(np.float32)
            finite = np.isfinite(x)
            if not finite.any():
                raise ValueError(f"{path}: CSIamp has no finite values")
            if not finite.all():
                x[~finite] = x[finite].mean(dtype=np.float64)
            frames[path] = x
        return np.stack([frames[path] for path in self._frames[index]])


def load_dataset(root, *, normalization="none"):
    return MMFiDataset(root, normalization)


if __name__ == "__main__":
    run_generator(load_dataset, dataset_name=MMFiDataset.name)
