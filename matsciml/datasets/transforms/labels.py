# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations
from typing import Callable, Literal
from logging import getLogger

import torch
from torch.utils.data import Dataset

from matsciml.common.types import DataDict
from matsciml.datasets.transforms.base import AbstractDataTransform
from matsciml.datasets.base import BaseLMDBDataset

logger = getLogger(__file__)


class AbstractLabelTransform(AbstractDataTransform):
    def __init__(
        self,
        label_key: str,
        value: float | torch.Tensor | None = None,
        num_samples: int | float | None = None,
    ) -> None:
        super().__init__()
        self.label_key = label_key
        self.value = value
        self.num_samples = num_samples

    def setup_transform(self, dataset: BaseLMDBDataset) -> None:
        """
        Hook to run as the transform is being initialized by a dataset.

        This allows the transform to obtain information about the dataset,
        including the class/type and computes a SHA512 hash based on samples.

        Parameters
        ----------
        dataset : BaseLMDBDataset
            Instance of an LMDB dataset.
        """
        # stash the dataset class name
        self.parent_dataset_type = dataset
        # we need to hash the dataset that's actually loaded from disk
        self.data_sha512 = dataset.data_sample_hash
        # convert fractional samples to actual number
        if isinstance(self.num_samples, float):
            assert (
                0.0 < self.num_samples <= 1.0
            ), "Fractional number of samples requested, but not between [0,1]."
            self.num_samples = int(self.num_samples * len(dataset))
        elif not self.num_samples:
            logger.warning(
                "Number of samples was not provided, "
                "statistics will be aggregated over the whole dataset!"
            )
        return super().setup_transform(dataset)

    @property
    def parent_dataset_type(self) -> str:
        """Return the name of the dataset type."""
        return self._parent_dataset_type

    @parent_dataset_type.setter
    def parent_dataset_type(self, parent: Dataset) -> None:
        self._parent_dataset_type = parent.__class__.__name__

    @property
    def data_sha512(self) -> str:
        """Return the pre-computed SHA-512 hash for the dataset."""
        return self._data_sha512

    @data_sha512.setter
    def data_sha512(self, value: str) -> None:
        self._data_sha512 = value

    @property
    def serializable_format(self) -> dict[str, str | float | None]:
        return {
            "dataset": self.parent_dataset_type,
            "sha512": self.data_sha512,
            "key": self.label_key,
            "value": getattr(self, "value", None),
            "agg_func": getattr(self, "agg_func", None),
            "num_samples": self.num_samples,
        }

    def sample_data(self) -> list[DataDict]:
        ...

    def compute_statistic(self, key: str, agg_func: Literal["mean", "std"] | Callable):
        ...
