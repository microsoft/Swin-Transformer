import concurrent.futures
import os
import pathlib
import warnings

import cv2
import einops
import timm.data
import torch
import torchvision
from tqdm.auto import tqdm


def load_statistics(directory):
    """
    Need to calculate mean and std for the individual channels so we can normalize the images.
    """
    dataset = timm.data.ImageDataset(
        root=directory, transform=torchvision.transforms.ToTensor()
    )
    channels, height, width = dataset[0][0].shape

    total = torch.zeros((channels,))
    total_squared = torch.zeros((channels,))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=32)

    for batch, _ in tqdm(dataloader):
        total += einops.reduce(batch, "batch channel height width -> channel", "sum")
        total_squared += einops.reduce(
            torch.mul(batch, batch), "batch channel height width -> channel", "sum"
        )

    divisor = len(dataset) * width * height

    mean = total / divisor
    var = total_squared / divisor - torch.mul(mean, mean)
    std = torch.sqrt(var)

    return mean, std


class Inat21:
    val_tar_gz_hash = "f6f6e0e242e3d4c9569ba56400938afc"
    train_tar_gz_hash = "e0526d53c7f7b2e3167b2b43bb2690ed"
    train_mini_tar_gz_hash = "db6ed8330e634445efc8fec83ae81442"
    strategies = ("resize", "pad")
    stages = ("train", "val", "train_mini")

    def __init__(self, root: str, stage: str, strategy: str, size: int):
        self.root = root
        self.stage = stage
        self._check_stage(stage)

        self.strategy = strategy
        self._check_strategy(strategy)

        self.size = size
        self._check_size(size)

    @property
    def directory(self) -> str:
        return os.path.join(self.root, f"{self.strategy}-{self.size}", self.stage)

    @property
    def ffcv(self) -> str:
        return os.path.join(
            self.root, f"{self.stage}-{self.strategy}-{self.size}.beton"
        )

    @property
    def tar_file(self):
        return os.path.join(self.root, "compressed", f"{self.stage}.tar.gz")

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw", self.stage)

    def _check_stage(self, stage):
        if stage not in self.stages:
            raise ValueError(f"Stage '{stage}' must be one of {self.stages}")

    def _check_strategy(self, strategy):
        if strategy not in self.strategies:
            raise ValueError(f"Strategy '{strategy}' must be one of {self.strategies}")

    def _check_size(self, size):
        if not isinstance(size, int):
            raise ValueError(f"Size {size} must be int; not {type(int)}")

    def check(self):
        # If <directory>/.finished doesn't exist, we need to preprocess.
        if not os.path.isfile(finished_file_path(self.directory)):
            warnings.warn(
                f"Data not processed in {self.directory}! "
                "You should run:\n\n"
                f"\tpython -m src.data preprocess {self.root} {self.stage} {self.strategy} {self.size}"
                "\n\nAnd then run this script again!"
            )
            raise RuntimeError(f"Data {self.directory} not pre-processed!")


def finished_file_path(directory):
    return os.path.join(directory, ".finished")


def preprocess_class(cls_dir, output_dir, strategy, size):
    cls = os.path.basename(cls_dir)
    output_dir = os.path.join(output_dir, cls)
    os.makedirs(output_dir, exist_ok=True)

    with os.scandir(cls_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue

            im = cv2.imread(entry.path)
            im = preprocess_image(im, strategy, size)
            output_path = os.path.join(output_dir, entry.name)
            if not cv2.imwrite(output_path, im):
                raise RuntimeError(output_path)


def preprocess_image(im, strategy, size):
    if strategy == "resize":
        return cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    elif strategy == "pad":
        # https://stackoverflow.com/questions/43391205/add-padding-to-images-to-get-them-into-the-same-shape
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def parent_of(path: str):
    return pathlib.Path(path).parents[0]


def preprocess_dataset(root: str, stage: str, strategy: str, size: int) -> None:
    inat = Inat21(root, stage, strategy, size)

    err_msg = (
        f"Can't prepare data for stage {stage}, strategy {strategy} and size {size}."
    )

    # 1. If the directory does not exist, ask the user to fix that for us.
    if not os.path.isdir(inat.raw_dir):
        # Check that the tar exists
        if not os.path.isfile(inat.tar_file):
            warn_msg = f"Please download the appropriate .tar.gz file to {root} for stage {stage}."
            if "raw" in root:
                warn_msg += f"\n\nYour root path should contain a 'raw' directory; did you mean to use {parent_of(root)}?\n"
            elif "compressed" in root:
                warn_msg += f"\n\nYour root path should contain a 'compressed' directory; did you mean to use {parent_of(root)}?\n"

            warnings.warn(warn_msg)

            raise RuntimeError(err_msg)
        else:
            warnings.warn(
                f"Please untar {inat.tar_file} in {root}. Probably need to run 'cd {root}; tar -xvf {inat.tar_file}"
            )
            raise RuntimeError(err_msg)

    # 2. Now that we know the raw directory exists, we need to convert it
    # to a processed directory
    out_path = os.path.join(root, inat.directory)

    # 3. Make sure the directory exists
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # 4. For all raw files, process and save to directory. We do this with
    # a process pool because it is both I/O (read/write) and CPU (image processing)
    # bound.
    with os.scandir(inat.raw_dir) as entries:
        directories = [entry.path for entry in entries if entry.is_dir()]
        # print(directories)
        print(f"Found {len(directories)} directories to preprocess.")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(preprocess_class, directory, out_path, strategy, size)
            for directory in tqdm(directories)
        ]

        print(f"Submitted {len(futures)} jobs to executor.")

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    # 5. Save a sentinel file called .finished
    open(finished_file_path(out_path), "w").close()
