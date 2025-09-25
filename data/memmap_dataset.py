"""Dataset helpers leveraging memory-mapped storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .mask_utils import MaskGenerator

__all__ = ["MaskGenerator", "MemmapImageDataset"]


def _load_metadata(path: Path) -> dict:
    """Load metadata describing a memmap-backed dataset.

    The helper supports ``.json``, ``.npy`` and ``.npz`` containers. JSON
    metadata is expected to be a mapping. ``.npy`` files must contain a mapping
    object (e.g. produced via ``np.save`` on a ``dict``), while ``.npz`` files
    are converted to a dictionary of Python lists.
    """

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if suffix == ".npy":
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            data = data.item()
        if not isinstance(data, dict):
            raise ValueError(f"Metadata in {path} must decode to a mapping, got {type(data)!r}.")
        return data
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            return {k: archive[k].tolist() for k in archive.files}
    raise ValueError(f"Unsupported metadata extension: {path.suffix}")


def _guess_metadata_path(memmap_path: Path) -> Path:
    """Infer the metadata file associated with ``memmap_path``."""

    candidates: Iterable[Path]
    if memmap_path.suffix:
        base = memmap_path.with_suffix("")
        suffix = memmap_path.suffix
        candidates = (
            memmap_path.with_suffix(suffix + ".meta.json"),
            memmap_path.with_suffix(suffix + ".meta.npy"),
            memmap_path.with_suffix(suffix + ".meta.npz"),
            base.with_suffix(".json"),
            base.with_suffix(".npy"),
            base.with_suffix(".npz"),
        )
    else:
        candidates = (
            memmap_path.with_suffix(".meta.json"),
            memmap_path.with_suffix(".meta.npy"),
            memmap_path.with_suffix(".meta.npz"),
            memmap_path.with_suffix(".json"),
            memmap_path.with_suffix(".npy"),
            memmap_path.with_suffix(".npz"),
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate metadata for {memmap_path}")


def _normalise_index(index: Any, dataset_length: int) -> List[Tuple[Any, ...]]:
    """Normalise index metadata to a list of tuples understood by ``numpy``."""

    if index is None:
        return [(i,) for i in range(dataset_length)]

    if isinstance(index, (str, bytes, Path)):
        index_array = np.load(index, allow_pickle=True)
    else:
        index_array = index

    if isinstance(index_array, np.ndarray):
        items = index_array.tolist()
    else:
        items = list(index_array)

    normalised: List[Tuple[Any, ...]] = []
    for entry in items:
        if isinstance(entry, (list, tuple, np.ndarray)):
            if isinstance(entry, np.ndarray):
                entry = entry.tolist()
            normalised.append(tuple(entry))
        else:
            normalised.append((entry,))
    return normalised


def _to_pil_image(array: np.ndarray) -> Image.Image:
    """Convert a numpy array to a PIL image with best-effort heuristics."""

    if array.ndim not in (2, 3):
        raise ValueError(f"Expected an image-like array with 2 or 3 dimensions, got shape {array.shape}.")

    arr = np.array(array)

    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)

    mode: Optional[str] = None
    if arr.ndim == 2:
        mode = "L"
    elif arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
            mode = "L"
        elif arr.shape[-1] == 3:
            mode = "RGB"

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).round().astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    return Image.fromarray(arr, mode=mode)


class MemmapImageDataset(Dataset):
    """Image dataset backed by ``numpy.memmap`` storage.

    Parameters
    ----------
    memmap_path:
        Path to the ``.memmap`` file containing the pixel data.
    transform:
        Optional callable applied to the loaded image. ``SimMIMTransform``
        expects the dataset to yield ``(img, mask)`` tuples, so the transform is
        typically responsible for producing the mask in addition to converting
        the input to tensors.
    metadata_path:
        Optional path to a metadata file. If omitted, :func:`_guess_metadata_path`
        attempts to locate an adjacent file describing the memmap. Metadata must
        contain ``shape`` (the global array shape) and ``dtype`` (the numpy data
        type). ``index`` is optional and, if present, should be an iterable where
        each element describes how to slice the memmap to retrieve a sample.
    """

    def __init__(
        self,
        memmap_path: Union[str, Path],
        transform: Optional[Callable] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.memmap_path = Path(memmap_path)
        if metadata_path is None:
            metadata_path = _guess_metadata_path(self.memmap_path)
        self.metadata_path = Path(metadata_path)

        metadata = _load_metadata(self.metadata_path)
        try:
            shape = tuple(int(v) for v in metadata["shape"])
            dtype = np.dtype(metadata["dtype"])
        except KeyError as exc:
            raise KeyError(
                f"Metadata in {self.metadata_path} is missing required key: {exc.args[0]!r}"
            ) from exc

        self._array_shape = shape
        self._dtype = dtype
        self._index = _normalise_index(metadata.get("index"), dataset_length=shape[0])
        self.transform = transform

        # Lazily created memmap handle to avoid pickling issues with DataLoader
        # workers using the ``spawn`` start method.
        self._memmap: Optional[np.memmap] = None

    def _ensure_memmap(self) -> np.memmap:
        if self._memmap is None:
            self._memmap = np.memmap(
                self.memmap_path, dtype=self._dtype, mode="r", shape=self._array_shape
            )
        return self._memmap

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._index)

    def __getitem__(self, idx: int):  # type: ignore[override]
        memmap = self._ensure_memmap()
        slice_spec = self._index[idx]

        if len(slice_spec) == 1:
            array = memmap[slice_spec[0]]
        else:
            array = memmap[tuple(slice_spec)]

        image = _to_pil_image(array)

        if self.transform is not None:
            return self.transform(image)
        return image

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # memmap objects cannot be pickled reliably; reopen in the worker.
        state["_memmap"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

