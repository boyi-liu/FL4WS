"""Load the official ARIL amplitude splits for activity recognition."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.io import loadmat

if __package__:
    from .utils import SampleMetadata, SensingDataset, run_generator
else:
    from utils import SampleMetadata, SensingDataset, run_generator


@dataclass(frozen=True)
class ARILSampleMetadata(SampleMetadata):
    location_id: Optional[int] = None


class ARILDataset(SensingDataset):
    name = "aril"
    classes = ("hand up", "hand down", "hand left", "hand right", "hand circle", "hand cross")

    def __init__(self, root, normalization):
        super().__init__(root, normalization)
        data_dir = self.root
        if not (data_dir / "train_data_split_amp.mat").is_file():
            data_dir = self.root / "data"

        self._arrays = {}
        rows, metadata = [], []
        for split in ("train", "test"):
            path = data_dir / f"{split}_data_split_amp.mat"
            feature_key = f"{split}_data"
            activity_key = f"{split}_activity_label"
            location_key = f"{split}_location_label"
            values = loadmat(path, variable_names=[feature_key, activity_key, location_key])
            if not {feature_key, activity_key, location_key} <= values.keys():
                raise ValueError(f"{path}: missing activity, location, or amplitude array")
            x = values[feature_key]
            activities = values[activity_key].reshape(-1)
            locations = values[location_key].reshape(-1)
            if x.ndim != 3 or x.shape[1:] != (52, 192) or np.iscomplexobj(x) or not len(x):
                raise ValueError(f"{path}: expected a nonempty real array of shape (N, 52, 192)")
            if len(x) != len(activities) or len(x) != len(locations):
                raise ValueError(f"{path}: feature and label counts differ")
            if not np.isin(activities, range(len(self.classes))).all() or not np.isin(locations, range(16)).all():
                raise ValueError(f"{path}: invalid activity or location label")
            self._arrays[split] = x.astype(np.float32)
            for row, (activity, location) in enumerate(zip(activities, locations)):
                rows.append((split, row))
                metadata.append(ARILSampleMetadata(
                    f"{split}/{row}", int(activity), source_split=split, location_id=int(location),
                ))

        self._rows = tuple(rows)
        self.metadata = tuple(metadata)

    def _read(self, index):
        split, row = self._rows[index]
        return self._arrays[split][row]


def load_dataset(root, *, normalization="none"):
    return ARILDataset(root, normalization)


if __name__ == "__main__":
    run_generator(load_dataset, dataset_name=ARILDataset.name)
