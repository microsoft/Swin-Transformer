import argparse

from .. import hierarchical
from . import datasets


def preprocess_cli(args):
    datasets.preprocess_dataset(args.root, args.stage, args.strategy, args.size)


def normalize_cli(args):
    mean, std = datasets.load_statistics(args.directory)
    print("Add this to a constants.py file:")
    print(
        f"""
"{args.directory}": (
    torch.tensor({mean.tolist()}),
    torch.tensor({std.tolist()}),
),"""
    )


def num_classes_cli(args):
    dataset = hierarchical.HierarchicalImageFolder(
        args.directory,
    )

    print("Add this to your config .yml file to the model section:")
    print(f"num_classes: {list(dataset.num_classes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Available commands.")

    # Preprocess
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess data.")
    preprocess_parser.add_argument(
        "root",
        help="Root data folder. Should contain folders compressed/ and raw/train, raw/val, etc.",
    )
    preprocess_parser.add_argument(
        "stage", choices=datasets.Inat21.stages, help="Data stage to preprocess"
    )
    preprocess_parser.add_argument(
        "strategy", choices=datasets.Inat21.strategies, help="Preprocessing strategy"
    )
    preprocess_parser.add_argument("size", type=int, help="Image size in pixels")
    preprocess_parser.set_defaults(func=preprocess_cli)

    # Normalize
    normalize_parser = subparsers.add_parser(
        "normalize", help="Measure mean and std of dataset."
    )
    normalize_parser.add_argument("directory", help="Data folder")
    normalize_parser.set_defaults(func=normalize_cli)

    # Number of classes
    num_classes_parser = subparsers.add_parser(
        "num-classes", help="Measure number of classes in dataset."
    )
    num_classes_parser.add_argument("directory", help="Data folder")
    num_classes_parser.set_defaults(func=num_classes_cli)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_usage()
