"""Load NTU-Fi HAR WiFi CSI and generate its federated data partitions."""

if __package__:
    from .utils import SampleMetadata, SensingDataset, read_csi_amplitude, run_generator
else:
    from utils import SampleMetadata, SensingDataset, read_csi_amplitude, run_generator


class NTUFiHARDataset(SensingDataset):
    name = "ntu-fi-har"

    def __init__(self, root, normalization):
        super().__init__(root, normalization)
        paths = {}
        for split in ("train", "test"):
            paths[split] = sorted(self.root.glob(f"{split}_amp/*/*.mat"))
            if not paths[split]:
                raise ValueError(f"{self.root}: no MAT samples found in {split}_amp/<class>/")
        self.classes = tuple(sorted({path.parent.name for group in paths.values() for path in group}))
        self._paths = tuple(path for group in paths.values() for path in group)
        self.metadata = tuple(
            SampleMetadata(
                path.relative_to(self.root).as_posix(),
                self.classes.index(path.parent.name),
                source_split=split,
            )
            for split, group in paths.items()
            for path in group
        )

    def _read(self, index):
        path = self._paths[index]
        amplitude = read_csi_amplitude(path)
        downsampled = amplitude[:, ::4]
        if downsampled.shape != (342, 500):
            raise ValueError(f"{path}: CSIamp shape {amplitude.shape} does not downsample to (342, 500)")
        return downsampled.reshape(3, 114, 500)


def load_dataset(root, *, normalization="none"):
    return NTUFiHARDataset(root, normalization)


if __name__ == "__main__":
    run_generator(load_dataset, dataset_name=NTUFiHARDataset.name)
