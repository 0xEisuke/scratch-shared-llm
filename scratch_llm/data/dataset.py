import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class ChunkedTokenDataset(Dataset):
    """Reads fixed-length token chunks produced by `prepare_data.py` via memmap."""

    def __init__(self, data_dir: str, split: str):
        with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)

        self.seq_len = meta["seq_len"]
        self.chunk_len = meta["chunk_len"]
        dtype = np.dtype(meta["dtype"])

        bin_path = os.path.join(data_dir, f"{split}.bin")
        arr = np.memmap(bin_path, dtype=dtype, mode="r")
        n_chunks = len(arr) // self.chunk_len
        self.data = arr[: n_chunks * self.chunk_len].reshape(n_chunks, self.chunk_len)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        chunk = torch.from_numpy(self.data[idx].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
