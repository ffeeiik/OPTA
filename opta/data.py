"""Small DataProto compatibility layer used by the OPTA evaluator.

The original project used VERL's DataProto container.  OPTA only needs a tiny
subset of that interface for offline evaluation: tensor batches, non-tensor
metadata, and concatenation of one or more trajectory outputs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class DataProto:
    batch: dict = field(default_factory=dict)
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, batch: dict) -> "DataProto":
        return cls(batch=dict(batch), non_tensor_batch={}, meta_info={})

    @classmethod
    def concat(cls, items: list["DataProto"]) -> "DataProto":
        if not items:
            return cls()
        if len(items) == 1:
            return copy.deepcopy(items[0])

        out = cls()
        batch_keys = set().union(*(item.batch.keys() for item in items))
        for key in batch_keys:
            values = [item.batch[key] for item in items if key in item.batch]
            if values and all(isinstance(value, torch.Tensor) for value in values):
                out.batch[key] = torch.cat(values, dim=0)
            else:
                out.batch[key] = values

        nt_keys = set().union(*(item.non_tensor_batch.keys() for item in items))
        for key in nt_keys:
            values = [item.non_tensor_batch[key] for item in items if key in item.non_tensor_batch]
            if values and all(isinstance(value, np.ndarray) for value in values):
                out.non_tensor_batch[key] = np.concatenate(values, axis=0)
            else:
                out.non_tensor_batch[key] = np.array(values, dtype=object)

        out.meta_info = copy.deepcopy(items[0].meta_info)
        return out

