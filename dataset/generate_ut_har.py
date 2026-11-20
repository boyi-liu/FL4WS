"""Load UT-HAR WiFi CSI and generate its federated data partitions."""

import numpy as np

if __package__:
    from .utils import SampleMetadata, SensingDataset, run_generator
else:
    from utils import SampleMetadata, SensingDataset, run_generator


class UTHARDataset(SensingDataset):
    name = "ut-har"

    def __init__(self, root, normalization):
        super().__init__(root, normalization)
        self._arrays = {}
        self._paths = {}
        raw_labels = {}
        for split in ("train", "val", "test"):
            path = self.root / "data" / f"X_{split}.csv"
            labels = np.load(self.root / "label" / f"y_{split}.csv", allow_pickle=False)
            if labels.ndim not in (1, 2) or (labels.ndim == 2 and 1 not in labels.shape):
                raise ValueError(f"UT-HAR {split}: expected a vector of labels")
            labels = labels.reshape(-1)
            if not labels.size or not np.isfinite(labels).all() or np.any(labels % 1):
                raise ValueError(f"UT-HAR {split}: expected nonempty integer labels")
            # The SenseFi release uses NPY data under .csv filenames.
            features = np.load(path, mmap_mode="r", allow_pickle=False)
            if features.ndim < 2 or len(features) != len(labels):
                raise ValueError(f"{path}: sample count does not match labels")
            if features.size != len(labels) * 250 * 90:
                raise ValueError(f"{path}: expected 250 × 90 values per sample")
            self._arrays[split] = features
            self._paths[split] = path
            raw_labels[split] = labels

        source_ids = sorted({int(label) for labels in raw_labels.values() for label in labels})
        self.classes = tuple(map(str, source_ids))
        label_ids = {source_id: idx for idx, source_id in enumerate(source_ids)}
        self._rows = tuple(
            (split, row) for split, labels in raw_labels.items() for row in range(len(labels))
        )
        self.metadata = tuple(
            SampleMetadata(f"{split}/{row}", label_ids[raw_labels[split][row]], source_split=split)
            for split, row in self._rows
        )

    def _read(self, index):
        split, row = self._rows[index]
        if split not in self._arrays:
            self._arrays[split] = np.load(self._paths[split], mmap_mode="r", allow_pickle=False)
        return self._arrays[split][row].reshape(1, 250, 90)

    def __getstate__(self):
        # Spawned workers reopen the memory maps instead of copying their contents.
        return {**self.__dict__, "_arrays": {}}


def load_dataset(root, *, normalization="none"):
    return UTHARDataset(root, normalization)


if __name__ == "__main__":
    run_generator(load_dataset, dataset_name=UTHARDataset.name)
