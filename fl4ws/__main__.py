"""Command-line interface for preparing a federated partition."""

from dataset import DATASETS, load_dataset
from dataset.utils import run_generator


def main(argv=None):
    run_generator(load_dataset, argv, datasets=DATASETS)


if __name__ == "__main__":
    main()
