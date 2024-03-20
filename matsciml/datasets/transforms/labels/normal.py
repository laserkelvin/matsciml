# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations
from typing import Literal
from logging import getLogger

import torch

from matsciml.common.types import DataDict
from matsciml.datasets.transforms.labels.base import AbstractLabelTransform


logger = getLogger(__file__)


class NormalLabelTransform(AbstractLabelTransform):
    def __init__(
        self,
        label_key: str,
        mean: float | torch.Tensor | None = None,
        std: float | torch.Tensor | None = None,
        value: float | torch.Tensor | None = None,
        agg_method: Literal["static", "sampled", "moving"] = "moving",
        num_samples: int | float | None = None,
    ) -> None:
        if value is not None:
            logger.info(
                "Value passed to NormalLabelTransform, but is not used."
                " Please ensure mean/std is passed instead."
            )
        super().__init__(label_key, value, agg_method, num_samples)
        self.mean = mean
        self.std = std

    def apply_transformation(self, sample: DataDict) -> DataDict:
        data = sample.get(self.label_key, None)
        if data is None:
            raise KeyError(f"Specified {self.label_key} did not return a value/tensor!")
        data -= self.mean
        data /= self.std
        sample[self.label_key] = data
        return sample

    def _static_agg_func(self, sample: DataDict) -> None:
        logger.info("No static agg function is implemented for normal distribution.")
        return None

    def _moving_agg_func(self, sample: DataDict) -> None:
        """
        Compute a moving mean and standard deviation.

        Algorithm is from Knuth, Art of Computer Programming, Vol 2.

        Parameters
        ----------
        sample : DataDict
            A single data sample used to retrieve a new
            label value.

        Raises
        ------
        KeyError:
            If we are unable to grab the specified label key from data.
        """
        # increment the moving average counter
        self.num_samples += 1
        new_value = sample.get(self.label_key, None)
        if new_value is None:
            raise KeyError("Specified label key is missing from data sample.")
        new_mean = self.mean + (new_value - self.mean) / self.num_samples
        # prevent underflow with some epsilon value
        new_var = max(
            self.std**2.0 + (new_value - self.mean) * (new_value - new_mean), 1e-7
        )
        new_std = new_var**0.5
        # update the actual values
        self.mean = new_mean
        self.std = new_std
        return None
