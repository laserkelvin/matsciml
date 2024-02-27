from __future__ import annotations

import pytorch_lightning as pl

from matsciml.datasets.transforms import (
    PointCloudToGraphTransform,
    PeriodicPropertiesTransform,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models.base import ScalarRegressionTask
from matsciml.models.pyg import EGNN
from matsciml.lightning import xpu  # noqa: F401

"""
This script demonstrates the use of Intel Data Center GPUs, running
a small EGNN model.

For the most part, this code will resemble the regular model demo
code, but the two modifications are:

1. Running `from matsciml.lightning import xpu` will ensure Lightning
recognizes XPU in terms of strategy and precision.
2. Adding "xpu_single" as the `Trainer` strategy will use a single tile/
device to execute on.
"""

# construct IS2RE relaxed energy regression with PyG implementation of E(n)-GNN
task = ScalarRegressionTask(
    encoder_class=EGNN,
    encoder_kwargs={"hidden_dim": 128, "output_dim": 64},
    task_keys=["energy_relaxed"],
)
# matsciml devset for OCP are serialized with DGL - this transform goes between the two frameworks
dm = MatSciMLDataModule.from_devset(
    "IS2REDataset",
    dset_kwargs={
        "transforms": [
            PeriodicPropertiesTransform(6.0, adaptive_cutoff=True),
            PointCloudToGraphTransform(
                "pyg",
                node_keys=["pos", "atomic_numbers"],
            ),
        ],
    },
)

# run a quick training loop
trainer = pl.Trainer(fast_dev_run=10, strategy="xpu_single")
trainer.fit(task, datamodule=dm)
