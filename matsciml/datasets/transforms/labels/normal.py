# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations
from typing import Literal, Any
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

    def _static_agg_func(self) -> Any:
        logger.info("No static agg function is implemented for normal distribution.")
        return None
