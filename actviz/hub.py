"""Fetching checkpoints from the Hugging Face Hub and reading safetensors.

The safetensors reader is hand-rolled (the format is an 8-byte header length,
a JSON header, then raw little-endian tensor bytes) so that bf16 checkpoints,
which numpy has no dtype for, can be read without pulling in torch.
"""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path

import numpy as np

_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,  # widened to float32 in SafetensorsFile.get
    "I64": np.int64,
    "I32": np.int32,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def download(repo: str, filename: str, revision: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, revision=revision))


class SafetensorsFile:
    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
            self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        header.pop("__metadata__", None)
        self._data_start = 8 + header_len
        self.header = header

    def keys(self):
        return self.header.keys()

    def get(self, name: str) -> np.ndarray:
        info = self.header[name]
        start, end = info["data_offsets"]
        raw = np.frombuffer(
            self._mm,
            dtype=_DTYPES[info["dtype"]],
            count=(end - start) // np.dtype(_DTYPES[info["dtype"]]).itemsize,
            offset=self._data_start + start,
        )
        if info["dtype"] == "BF16":
            raw = (raw.astype(np.uint32) << 16).view(np.float32)
        return raw.reshape(info["shape"])


class Checkpoint:
    """All tensors of a (possibly sharded) safetensors checkpoint, read lazily, as float32."""

    def __init__(self, files: list[SafetensorsFile]):
        self._where = {k: f for f in files for k in f.keys()}

    @classmethod
    def from_hub(cls, repo: str, revision: str | None = None) -> "Checkpoint":
        try:
            return cls([SafetensorsFile(download(repo, "model.safetensors", revision))])
        except Exception as single_err:
            try:
                index = json.loads(download(repo, "model.safetensors.index.json", revision).read_text())
            except Exception:
                raise single_err
            shards = sorted(set(index["weight_map"].values()))
            return cls([SafetensorsFile(download(repo, s, revision)) for s in shards])

    def __contains__(self, name: str) -> bool:
        return name in self._where

    def __getitem__(self, name: str) -> np.ndarray:
        return np.array(self._where[name].get(name), dtype=np.float32)
