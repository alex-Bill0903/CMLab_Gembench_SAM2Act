

# From: https://github.com/stepjam/YARR/blob/main/yarr/replay_buffer/wrappers/pytorch_replay_buffer.py

import time
from threading import Lock, Thread

import os
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader

from yarr.replay_buffer.replay_buffer import ReplayBuffer
from yarr.replay_buffer.wrappers import WrappedReplayBuffer
import time


class PyTorchIterableReplayDataset(IterableDataset):

    def __init__(self, replay_buffer: ReplayBuffer, sample_mode, sample_distribution_mode = 'transition_uniform'):
        self._replay_buffer = replay_buffer
        self._sample_mode = sample_mode
        if self._sample_mode == 'enumerate':
            self._num_samples = self._replay_buffer.prepare_enumeration()
        self._sample_distribution_mode = sample_distribution_mode

    def _generator(self):
        while True:
            if self._sample_mode == 'random':
                yield self._replay_buffer.sample_transition_batch(pack_in_dict=True, distribution_mode = self._sample_distribution_mode)
            elif self._sample_mode == 'enumerate':
                yield self._replay_buffer.enumerate_next_transition_batch(pack_in_dict=True)

    def __iter__(self):
        return iter(self._generator())

    def __len__(self): # enumeration will throw away the last incomplete batch
        return self._num_samples // self._replay_buffer._batch_size

# import numpy as np  
# class LangAugIterableDataset(IterableDataset):
#     def __init__(self, base_dataset):
#         self.base = base_dataset

#     def __iter__(self):
#         for sample in self.base:  # sample 是 dict，包含 "lang_goal_embs" 與 "lang_goal_texts"
#             embs = sample.pop("lang_goal_embs")       # shape = (n_desc, D)
#             texts = sample.pop("lang_goal_texts")     # shape = (n_desc,)
#             # randomly choose one of the n_desc
#             idx = np.random.randint(0, embs.shape[0])
#             # put it back to sample
#             sample["lang_goal_emb"]  = embs[idx]     # shape = (D,)
#             sample["lang_goal_text"] = texts[idx]    # str
#             yield sample

class PyTorchReplayBuffer(WrappedReplayBuffer):
    """Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism.

    Usage:
      To add a transition:  call the add function.

      To sample a batch:    Construct operations that depend on any of the
                            tensors is the transition dictionary. Every sess.run
                            that requires any of these tensors will sample a new
                            transition.
      sample_mode: the mode to sample data, choose from ['random', 'enumerate']
    """

    def __init__(self, replay_buffer: ReplayBuffer, num_workers: int = 2, sample_mode = 'random', sample_distribution_mode = 'transition_uniform'):
        super(PyTorchReplayBuffer, self).__init__(replay_buffer)
        self._num_workers = num_workers
        self._sample_mode = sample_mode
        self._sample_distribution_mode = sample_distribution_mode

    ## origianl code ###
    def dataset(self) -> DataLoader:
        d = PyTorchIterableReplayDataset(self._replay_buffer, self._sample_mode, self._sample_distribution_mode)

        # Batch size None disables automatic batching
        return DataLoader(d, batch_size=None, pin_memory=True,
                          num_workers=self._num_workers)
    ## origianl code ###
    
    # def dataset(self) -> DataLoader:
    #     # 1) original iterable dataset
    #     base_ds = PyTorchIterableReplayDataset(
    #         self._replay_buffer,
    #         sample_mode=self._sample_mode,
    #         sample_distribution_mode=self._sample_distribution_mode,
    #     )
    #     # 2) randomly choose one of the n_desc
    #     aug_ds = LangAugIterableDataset(base_ds)

    #     # 3) return DataLoader
    #     return DataLoader(
    #         aug_ds,
    #         batch_size=None,
    #         pin_memory=True,
    #         num_workers=self._num_workers,
    #     )
