from __future__ import annotations

import torch
import pytest

from matsciml.common.types import AtomicStructure


@pytest.fixture
def atomic_structure() -> AtomicStructure:
    NUM_ATOMS = 64
    pos = torch.rand(NUM_ATOMS, 3)
    atomic_numbers = torch.randint(1, 100, (NUM_ATOMS,))
    targets = {"energy": -215.0, "force": torch.randn_like(pos)}
    target_keys = {"regression": ["energy", "force"]}
    struct = AtomicStructure(
        pos=pos,
        atomic_numbers=atomic_numbers,
        targets=targets,
        dataset="TestDataset",
        target_keys=target_keys,
    )
    return struct


@pytest.mark.parametrize(
    "key",
    (
        "pos",
        "force",
        pytest.param("blah", marks=pytest.mark.xfail(reason="Key does not exist.")),
    ),
)
def test_contains_check(atomic_structure, key):
    """Check to make sure we we can retrieve tensors and fail on missing keys"""
    assert key in atomic_structure


@pytest.mark.parametrize(
    "key",
    (
        "pos",
        "energy",
        pytest.param("blah", marks=pytest.mark.xfail(reason="Key does not exist.")),
    ),
)
def test_get_item(atomic_structure, key):
    """Check to make sure we we can retrieve tensors and fail on missing keys"""
    assert atomic_structure[key] is not None


@pytest.mark.parametrize(
    "key",
    (
        "pos",
        pytest.param("energy", marks=pytest.mark.xfail(reason="Energy is scalar.")),
        "force",
        pytest.param(
            "atomic_numbers",
            marks=pytest.mark.xfail(reason="Only float tensors can have gradients"),
        ),
        pytest.param("blah", marks=pytest.mark.xfail(reason="Key does not exist.")),
    ),
)
def test_grad_toggle(atomic_structure, key):
    """Check to make sure we can set the require grad state"""
    tensors = atomic_structure.tensors
    atomic_structure.set_require_grad(key, True)
    assert tensors[key].requires_grad is True


@pytest.mark.parametrize(
    "dtype",
    (
        torch.float32,
        torch.float64,
        None,
        pytest.param(
            torch.bool, marks=pytest.mark.xfail(reason="Can't cast to boolean.")
        ),
    ),
)
@pytest.mark.parametrize(
    "device",
    (
        None,
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="No CUDA device."
            ),
        ),
        pytest.param(
            "xpu",
            marks=pytest.mark.skipif(
                not torch.xpu.is_available(), reason="No XPU device."
            ),
        ),
    ),
)
def test_data_movement(atomic_structure, dtype, device):
    atomic_structure.to(device, dtype)


@pytest.mark.parametrize("entry", (("cell", torch.rand(1, 3, 3)), ("energy", 50.0)))
def test_item_set(atomic_structure, entry):
    """Check that we can set a target or attribute to the class"""
    name, value = entry
    atomic_structure[name] = value
    if name == "energy":
        assert name in atomic_structure.targets
    else:
        assert name in atomic_structure
