"""Load the SenseFi Widar 3.0 BVP release from CSV files or its original ZIP."""

import os
from math import prod
from pathlib import PurePosixPath
import re
from zipfile import BadZipFile, ZipFile

import numpy as np

if __package__:
    from .utils import SampleMetadata, SensingDataset, run_generator
else:
    from utils import SampleMetadata, SensingDataset, run_generator


class Widar3Dataset(SensingDataset):
    name = "widar3"
    shape = (22, 20, 20)

    def __init__(self, root, normalization):
        super().__init__(root, normalization)
        zip_path = self.root / "Widardata.zip"
        self._zip_path = zip_path if zip_path.is_file() else None
        self._archive = None
        self._archive_pid = None

        if self._zip_path is not None:
            try:
                with ZipFile(self._zip_path) as archive:
                    paths = archive.namelist()
            except BadZipFile as error:
                raise ValueError(f"{self._zip_path}: invalid Widar ZIP archive") from error
        else:
            paths = [
                path.relative_to(self.root).as_posix()
                for split in ("train", "test")
                for path in self.root.glob(f"{split}/*/*.csv")
            ]

        rows = {}
        categories = {}
        for path in paths:
            parts = PurePosixPath(path).parts
            if parts[:1] == ("Widardata",):
                parts = parts[1:]
            if len(parts) != 3 or parts[0] not in {"train", "test"} or not parts[2].endswith(".csv"):
                continue
            split, category, filename = parts
            label = re.fullmatch(r"([0-9]+)-(.+)", category)
            if label is None:
                raise ValueError(f"{path}: expected a numbered gesture directory")
            code, gesture = int(label[1]), label[2]
            if categories.setdefault(code, gesture) != gesture:
                raise ValueError(f"{path}: gesture names disagree for label {code}")
            subject = re.match(r"(user[0-9]+)-", filename)
            if subject is None:
                raise ValueError(f"{path}: expected a Widar user identifier in the file name")
            sample_id = "/".join(parts)
            if sample_id in rows:
                raise ValueError(f"{path}: duplicate Widar sample")
            rows[sample_id] = (path, code, subject[1], split)

        if {row[3] for row in rows.values()} != {"train", "test"}:
            raise ValueError(f"{self.root}: expected Widardata.zip or train/<gesture> and test/<gesture> CSVs")
        label_ids = {code: idx for idx, code in enumerate(sorted(categories))}
        self.classes = tuple(categories[code] for code in label_ids)
        ordered_rows = sorted(rows.items())
        self._paths = tuple(path for _, (path, _, _, _) in ordered_rows)
        self.metadata = tuple(
            SampleMetadata(sample_id, label_ids[code], subject_id=subject, source_split=split)
            for sample_id, (_, code, subject, split) in ordered_rows
        )

    @property
    def preprocessing(self):
        return {**super().preprocessing, "representation": "body-coordinate velocity profile"}

    def _read(self, index):
        path = self._paths[index]
        if self._zip_path is not None:
            # Each worker needs its own file descriptor when a data loader forks.
            if self._archive is None or self._archive_pid != os.getpid():
                if self._archive is not None:
                    self._archive.close()
                self._archive = ZipFile(self._zip_path)
                self._archive_pid = os.getpid()
            stream = self._archive.open(path)
        else:
            stream = (self.root / path).open("rb")
        with stream:
            x = np.loadtxt(stream, delimiter=",", dtype=np.float32)
        if x.size != prod(self.shape):
            raise ValueError(f"{path}: expected {self.shape} BVP values")
        return x.reshape(self.shape)

    def __getstate__(self):
        return {**self.__dict__, "_archive": None, "_archive_pid": None}


def load_dataset(root, *, normalization="none"):
    return Widar3Dataset(root, normalization)


if __name__ == "__main__":
    run_generator(load_dataset, dataset_name=Widar3Dataset.name)
