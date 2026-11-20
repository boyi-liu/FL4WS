"""Prepare federated partitions or launch a FedAvg process."""

import argparse
from pathlib import Path
import sys

import yaml

from dataset import DATASETS, load_dataset
from dataset.utils import add_generator_arguments, run_partition_command
from utils._training import load_training_config


def _run_training(parser, args):
    if args.role == "client" and (args.client_id is None or args.client_id < 0):
        parser.error("--role client requires a nonnegative --client-id")
    if args.role == "server" and args.client_id is not None:
        parser.error("--client-id is only valid with --role client")

    try:
        config = load_training_config(args.config)
        if args.role == "server":
            from server.fedavg import FedAvgServer

            server = FedAvgServer(
                config.partition, rounds=config.rounds, client_fraction=config.client_fraction, seed=config.seed,
            )
            server.run(
                config.host, config.port, timeout=config.timeout, output=config.output,
                eval_every=config.eval_every, eval_batch_size=config.batch_size,
            )
        else:
            from client.fedavg import FedAvgClient

            client = FedAvgClient(
                config.partition, args.client_id, local_epochs=config.local_epochs,
                batch_size=config.batch_size, lr=config.lr, seed=config.seed,
            )
            client.run(config.host, config.port, timeout=config.timeout)
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        parser.exit(1, "Training requires PyTorch. Run `python -m pip install -e '.[models]'`.\n")
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as error:
        parser.exit(1, f"{args.role.capitalize()} error: {error}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fl4ws", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    partition = commands.add_parser("partition", help="generate federated data partitions")
    add_generator_arguments(partition, datasets=DATASETS)
    training = commands.add_parser("train", help="start a FedAvg server or client")
    training.add_argument("--config", required=True, type=Path, help="shared training YAML file")
    training.add_argument("--role", required=True, choices=("server", "client"))
    training.add_argument("--client-id", type=int, help="client ID from the partition manifest")

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv.insert(0, "partition")
    args = parser.parse_args(argv)
    if args.command == "train":
        _run_training(training, args)
    else:
        del args.command
        run_partition_command(load_dataset, partition, args)


if __name__ == "__main__":
    main()
