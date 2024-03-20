# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations
from typing import Callable, Literal
from logging import getLogger
from hashlib import sha512

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
        # we need to hash the dataset that's loaded
        data_sha = self._hash_dataset(dataset)
        self.data_sha512 = data_sha
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

    @staticmethod
    def _hash_dataset(dataset: BaseLMDBDataset) -> str:
        """
        Compute a hash for a dataset based on indicative samples.

        This function intends to summarize a given dataset based on
        five raw data samples from it. By drawing from start, end,
        and three percentile points in between, the idea is that
        we should hopefully capture the composition of a dataset
        even if it constitutes multiple LMDB files. That we can
        reliably say the statistics being used/computed actually
        correspond to the data being used.

        Parameters
        ----------
        dataset : BaseLMDBDataset
            Instance of an LMDB dataset. We reply on `index_to_key`
            and `data_from_key` methods to sample from every LMDB
            file being pointed to, as well as getting the data sample
            before any transformations respectively.

        Returns
        -------
        str
            SHA-512 hash based off five data samples from the dataset.
        """
        hasher = sha512()
        num_samples = len(dataset)
        # get the first and last points, plus quantiles in between
        # as indicative samples
        indices = (
            [
                0,
            ]
            + [int(num_samples * percent) for percent in [0.25, 0.5, 0.75]]
            + num_samples
        )
        logger.info(f"Hashing {dataset} at indices {indices}")
        for index in indices:
            (lmdb_index, subindex) = dataset.index_to_key(index)
            sample = dataset.data_from_key(lmdb_index, subindex)
            # convert data sample to binary format for hashing
            hasher.update(bytes(str(sample), "utf-8"))
        value = hasher.hexdigest()
        logger.info(f"Produced SHA512: {value}")
        return value

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
