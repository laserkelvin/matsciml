import pytest

import torch

from ocpmodels.lightning.data_utils import S2EFDGLDataModule
from ocpmodels.datasets import S2EFDataset, s2ef_devset
from ocpmodels.datasets import transforms


@pytest.mark.dependency()
def test_distance_transform():
    trans = [transforms.DistancesTransform()]
    dset = S2EFDataset(s2ef_devset, transforms=trans)
    batch = dset.__getitem__(0)
    assert "r" in batch.get("graph").edata


@pytest.mark.dependency(["test_distance_transform"])
def test_graph_variable_transform():
    trans = [transforms.DistancesTransform(), transforms.GraphVariablesTransform()]
    dset = S2EFDataset(s2ef_devset, transforms=trans)
    batch = dset.__getitem__(0)
    assert "graph_variables" in batch


@pytest.mark.dependency(["test_graph_variable_transform"])
def test_batched_gv_transform():
    trans = [transforms.DistancesTransform(), transforms.GraphVariablesTransform()]
    dm = S2EFDGLDataModule.from_devset(transforms=trans)
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert "graph_variables" in batch
    gv = batch.get("graph_variables")
    assert gv.ndim == 2
    assert gv.shape == (8, 9)
    assert torch.all(~torch.isnan(gv))


@pytest.mark.dependency()
def test_remove_tag_zero():
    trans = [
        transforms.RemoveTagZeroNodes(),
    ]
    dm = S2EFDGLDataModule.from_devset(transforms=trans)
    dm.setup()
    loader = dm.train_dataloader()
    graph = next(iter(loader))["graph"]
    # make sure we've purged all of the tag zero nodes
    assert not torch.any(graph.ndata["tags"] == 0)


@pytest.mark.dependency(["test_remove_tag_zero"])
def test_graph_supernode():
    trans = [transforms.GraphSupernode(100), transforms.RemoveTagZeroNodes()]
    dm = S2EFDGLDataModule.from_devset(transforms=trans)
    dm.setup()
    loader = dm.train_dataloader()
    graph = next(iter(loader))["graph"]
    # should be one super node per graph
    assert (graph.ndata["tags"] == 3).sum() == graph.batch_size
