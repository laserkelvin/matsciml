from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from matsciml.datasets import IS2REDataset
from matsciml.datasets.materials_project import MaterialsProjectDataset
from matsciml.datasets.multi_dataset import MultiDataset

# make the test deterministic
torch.manual_seed(21515)


@pytest.mark.dependency()
def test_joint_dataset():
    is2re = IS2REDataset.from_devset()
    mp = MaterialsProjectDataset.from_devset()

    joint = MultiDataset([is2re, mp])
    # try and grab a sample
    _ = joint.__getitem__(0)


@pytest.mark.dependency(depends=["test_joint_dataset"])
def test_joint_batching():
    is2re = IS2REDataset.from_devset()
    mp = MaterialsProjectDataset.from_devset()

    joint = MultiDataset([is2re, mp])
    # try and grab a sample
    loader = DataLoader(joint, batch_size=8, shuffle=False, collate_fn=joint.collate_fn)
    batch = next(iter(loader))
    assert "IS2REDataset" in batch


@pytest.mark.dependency(depends=["test_joint_batching"])
def test_joint_batching_shuffled():
    is2re = IS2REDataset.from_devset()
    mp = MaterialsProjectDataset.from_devset()

    joint = MultiDataset([is2re, mp])
    # try and grab a sample
    loader = DataLoader(joint, batch_size=8, shuffle=True, collate_fn=joint.collate_fn)
    batch = next(iter(loader))
    # check both datasets are in the batch
    assert all([key in batch for key in ["MaterialsProjectDataset", "IS2REDataset"]])


@pytest.mark.dependency(depends=["test_joint_dataset"])
def test_target_keys():
    is2re = IS2REDataset.from_devset()
    mp = MaterialsProjectDataset.from_devset()

    joint = MultiDataset([is2re, mp])
    keys = joint.target_keys
    expected_is2re = {"regression": ["energy_init", "energy_relaxed"]}
    expected_mp = {
        "classification": ["is_metal", "is_magnetic", "is_stable"],
        "regression": [
            "uncorrected_energy_per_atom",
            "efermi",
            "energy_per_atom",
            "band_gap",
            "formation_energy_per_atom",
        ],
    }

    for result_keys, expected_keys in [
        (keys["IS2REDataset"], expected_is2re),
        (keys["MaterialsProjectDataset"], expected_mp),
    ]:
        assert (
            result_keys.keys() == expected_keys.keys()
        ), f"Expected target key types {expected_keys.keys()}, got {result_keys.keys()}"

        for key, target_values in expected_keys.items():
            assert sorted(target_values) == sorted(
                result_keys[key]
            ), f"Expected target keys {target_values}, got {expected_keys[key]}"
