# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations
from typing import Any, Literal
from logging import getLogger
from abc import abstractmethod
from random import sample as choose_without_replacement

import torch
from torch.utils.data import Dataset

from matsciml.common.types import DataDict
from matsciml.datasets.transforms.base import AbstractDataTransform
from matsciml.datasets.base import BaseLMDBDataset

logger = getLogger(__file__)

__all__ = ["AbstractLabelTransform"]


class AbstractLabelTransform(AbstractDataTransform):
    __valid_agg_str__ = ["static", "sampled", "moving"]

    def __init__(
        self,
        label_key: str,
        value: float | torch.Tensor | None = None,
        agg_method: Literal["static", "sampled", "moving"] = "moving",
        num_samples: int | float | None = None,
    ) -> None:
        super().__init__()
        self.label_key = label_key
        self.value = value
        assert (
            agg_method in self.__valid_agg_str__
        ), f"Requested agg_method not valid; available: {self.__valid_agg_str__}"
        self.agg_method = agg_method
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
        self._dataset_len = len(dataset)
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

    @abstractmethod
    def _sampled_agg_func(self, *args, **kwargs) -> Any:
        """Implements a sampling-based version of the label transformation."""
        ...

    @abstractmethod
    def _static_agg_func(self, *args, **kwargs) -> Any:
        """Implements an immutable version of the label transformation."""
        ...

    @abstractmethod
    def _moving_agg_func(self, *args, **kwargs) -> Any:
        """Implements an moving/tracked version of the label transformation."""
        ...

    @abstractmethod
    def apply_transformation(self, sample: DataDict) -> DataDict:
        """Implements the actual transformation step."""
        ...

    def __call__(self, data: DataDict) -> DataDict:
        """Introduced to provide the same behavior as any other transform."""
        return self.apply_transformation(data)

    def sample_data(self, dataset: BaseLMDBDataset) -> list[DataDict]:
        """
        Draw random samples from a target dataset.

        This method will call ``dataset.__getitem__`` to retrieve a specified
        number of samples without replacement.

        Parameters
        ----------
        dataset : BaseLMDBDataset
            Instance of an LMDB dataset.

        Returns
        -------
        list[DataDict]
            List of data samples retrieved from the dataset.
        """
        assert self.num_samples, "Data sampling needs a specified `num_samples`!"
        indices = list(range(self._dataset_len))
        sample_indices = choose_without_replacement(indices, k=self.num_samples)
        data_samples = []
        for index in sample_indices:
            sample = dataset.__getitem__(index)
            data_samples.append(sample)
        return data_samples
