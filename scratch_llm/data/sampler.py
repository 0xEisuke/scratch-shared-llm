from typing import Iterator, List

import torch
from torch.utils.data import Sampler


class ResumableShuffleBatchSampler(Sampler[List[int]]):
    """Endless stream of shuffled, drop_last batches that can be re-entered
    at any batch position.

    The order depends only on (n_samples, batch_size, seed): epoch `e` is the
    `e`-th permutation drawn from a private generator seeded with `seed`, so
    it is unaffected by the global RNG (and therefore by how much randomness
    model init consumed). `start_batch` is the number of batches already
    consumed by a previous process; iteration continues with the batch an
    uninterrupted run would have produced next, instead of replaying the
    shuffle from the top.
    """

    def __init__(self, n_samples: int, batch_size: int, seed: int, start_batch: int = 0):
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.seed = seed
        self.start_batch = start_batch
        self.batches_per_epoch = n_samples // batch_size
        if self.batches_per_epoch == 0:
            raise ValueError(
                f"dataset has {n_samples} samples, fewer than one batch of {batch_size}"
            )

    def __iter__(self) -> Iterator[List[int]]:
        start_epoch, offset = divmod(self.start_batch, self.batches_per_epoch)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        # Replay the permutations of already-finished epochs so the generator
        # is in the same state an uninterrupted run would have reached.
        for _ in range(start_epoch):
            torch.randperm(self.n_samples, generator=generator)

        epoch = start_epoch
        while True:
            if epoch > 0 and offset == 0:
                print(
                    f"warning: starting epoch {epoch + 1} over this data loader (dataset is repeating)"
                )
            perm = torch.randperm(self.n_samples, generator=generator)
            for b in range(offset, self.batches_per_epoch):
                yield perm[b * self.batch_size : (b + 1) * self.batch_size].tolist()
            epoch += 1
            offset = 0
