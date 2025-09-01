# Utilities for Balanced MixUp technique
# Ported from: https://github.com/agaldran/balanced_mixup/blob/main/utils/get_loaders.py

from typing import Literal

import numpy as np
from torch import Tensor
from torch.utils.data import Sampler, WeightedRandomSampler

SamplingMode = Literal['instance', 'class', 'sqrt', 'cbrt']


def get_sampling_probabilities(
    class_count: np.ndarray | Tensor,
    mode: SamplingMode,
) -> np.ndarray | Tensor:
    if mode == 'instance':
        q = 0
    elif mode == 'class':
        q = 1
    elif mode == 'sqrt':
        q = 0.5  # 1/2
    elif mode == 'cbrt':
        q = 0.125  # 1/8
    else:
        raise ValueError(f'Unknown mode: {mode}')

    relative_freq = class_count**q / (class_count**q).sum()
    sampling_probabilities = relative_freq ** (-1)
    return sampling_probabilities


def get_data_loader_sampler(labels: list[int], mode: SamplingMode) -> Sampler[int]:
    class_count = np.unique(labels, return_counts=True)[1]
    sampling_probs = get_sampling_probabilities(class_count, mode=mode)
    sample_weights = sampling_probs[labels].tolist()

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler


# https://github.com/huanghoujing/pytorch-wrapping-multi-dataloaders/blob/master/wrapping_multi_dataloaders.py
class ComboIter(object):
    """An iterator."""

    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [next(loader_iter) for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)


class ComboDataLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
    loaders: a list or tuple of pytorch DataLoader objects
    """

    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return ComboIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches
