from typing import List, Union, Optional
from functools import partial

import torch

from ocpmodels.common import DataDict, package_registry
from ocpmodels.common.types import DataDict, GraphTypes, AbstractGraph
from ocpmodels.datasets.base import BaseLMDBDataset
from ocpmodels.datasets.transforms.representations import RepresentationTransform
from ocpmodels.datasets import utils

"""
Transforms that create point cloud representations from graphs.
"""

if package_registry["dgl"]:
    import dgl
    from dgl import DGLGraph

if package_registry["pyg"]:
    import torch_geometric
    from torch_geometric.data import Data as PyGGraph

__all__ = ["GraphToPointCloudTransform", "OCPGraphToPointCloudTransform"]


class GraphToPointCloudTransform(RepresentationTransform):
    def __init__(self, backend: str, full_pairwise: bool = True) -> None:
        """
        _summary_

        Parameters
        ----------
        backend : str
            Either 'dgl' or 'pyg'; specifies that graph framework to
            represent structures
        full_pairwise : bool, optional
            If True, creates atom-centered point clouds; by default True
        """
        super().__init__(backend=backend)
        self.full_pairwise = full_pairwise

    def setup_transform(self, dataset: BaseLMDBDataset) -> None:
        """
        This modifies the dataset's collate function by replacing it with
        a partial with pad_keys specified.

        Parameters
        ----------
        dataset : BaseLMDBDataset
            A dataset object which is a subclass of `BaseLMDBDataset`.
        """
        dataset.representation = "point_cloud"
        # we will pack point cloud features, but not positions
        collate_fn = partial(
            utils.concatenate_keys, pad_keys=["pc_features"], unpacked_keys=["pos"]
        )
        dataset.collate_fn = staticmethod(collate_fn).__func__
        return super().setup_transform(dataset)

    def prologue(self, data: DataDict) -> None:
        assert self._check_for_type(
            data, GraphTypes
        ), f"No graphs to transform into point clouds!"
        assert data["dataset"] in [
            "IS2REDataset",
            "S2EFDataset",
        ], f"Dataset not from OCP; this transform should only be applied to IS2RE/S2EF."
        return super().prologue(data)

    if package_registry["dgl"]:

        def _convert_dgl(self, g: dgl.DGLGraph, data: DataDict) -> None:
            assert isinstance(
                g, dgl.DGLGraph
            ), f"Expected DGL graph as input, but got {g} which is type {type(g)}"
            features = g.ndata["atomic_numbers"].long()
            system_size = len(features)
            src_indices = torch.arange(system_size)
            if not self.full_pairwise:
                num_neighbors = torch.randint(1, system_size, (1,)).item()
                # extract out a random number of neighbors and sort the indices
                dst_indices = torch.randperm(system_size)[:num_neighbors].sort().values
            else:
                dst_indices = src_indices
            pos = g.ndata["pos"]
            # extract out point cloud features
            features = utils.point_cloud_featurization(
                features[src_indices], features[dst_indices], 100
            )
            data["pos"] = pos  # left as N, 3
            data["pc_features"] = features
            data["sizes"] = system_size
            data["src_nodes"] = src_indices
            data["dst_nodes"] = dst_indices

    if package_registry["pyg"]:

        @staticmethod
        def _convert_pyg(g, data: DataDict) -> None:
            ...

    def convert(self, data: DataDict) -> None:
        graph: AbstractGraph = data["graph"]
        if self.backend == "dgl":
            self._convert_dgl(graph, data)
        else:
            self._convert_pyg(graph, data)

    def epilogue(self, data: DataDict) -> None:
        try:
            del data["graph"]
        except KeyError:
            pass
        return super().prologue(data)


class OCPGraphToPointCloudTransform(GraphToPointCloudTransform):
    def __init__(
        self, backend: str, sample_size: int = 5, full_pairwise: bool = True
    ) -> None:
        super().__init__(backend, full_pairwise)
        self.sample_size = sample_size

    @staticmethod
    def _extract_indices(tags: torch.Tensor, nodes: torch.Tensor) -> List[List[int]]:
        """
        Extract out indices of nodes, given a tensor containing the OCP
        tags. This is written as in a framework agnostic way, with the
        intention to be re-used between PyG and DGL workflows.

        Parameters
        ----------
        tags : torch.Tensor
            1D Long tensor containing 0,1,2 for each node corresponding
            to molecule, surface, and substrate nodes respectively
        nodes : torch.Tensor
            1D tensor of node IDs

        Returns
        -------
        List[List[int]]
            List of three lists, corresponding to node indices for
            each category of separated nodes
        """
        molecule_idx = nodes[[tags == 2]]
        surface_idx = nodes[[tags == 1]]
        substrate_idx = nodes[[tags == 0]]
        # get nodes out
        molecule_nodes = nodes[molecule_idx].tolist()
        surface_nodes = nodes[surface_idx].tolist()
        substrate_nodes = nodes[substrate_idx].tolist()
        return molecule_nodes, surface_nodes, substrate_nodes

    def _pick_src_dst(
        self,
        molecule_nodes: List[int],
        surface_nodes: List[int],
        substrate_nodes: List[int],
    ) -> List[List[int]]:
        """
        Separates nodes into source and destination, as part of creating a more
        compact, molecule/atom centered representation of the point cloud. For
        each atom that is marked as part of the adsorbate/molecule, our positions
        and featurization factors in pairwise interactions with other molecule atoms
        as well as a random sampling of surface/substrate atoms.

        Parameters
        ----------
        molecule_nodes : List[int]
            List of indices corresponding to nodes that constitute the adsorbate
        surface_nodes : List[int]
            List of indices corresponding to nodes that constitute the surface
        substrate_nodes : List[int]
            List of indices corresponding to nodes that constitute the substrate

        Returns
        -------
        src_nodes, dst_nodes
            List of node indices used for source/destination designation
        """
        num_samples = max(
            self.sample_size - len(molecule_nodes) + len(surface_nodes), 0
        )
        if isinstance(substrate_nodes, list):
            substrate_nodes = torch.tensor(substrate_nodes)
        neighbor_idx = substrate_nodes[
            torch.randperm(min(num_samples, len(substrate_nodes)))
        ].tolist()
        src_nodes = torch.LongTensor(molecule_nodes)
        dst_nodes = torch.LongTensor(molecule_nodes + surface_nodes + neighbor_idx)
        # in the full pairwise, make the point cloud neighbors symmetric
        if self.full_pairwise:
            src_nodes = dst_nodes
        return src_nodes, dst_nodes

    if package_registry["dgl"]:

        def _convert_dgl(self, g: DGLGraph, data: DataDict) -> None:
            tags, nodes, atomic_numbers = (
                g.ndata["tags"],
                g.nodes(),
                g.ndata["atomic_numbers"],
            )
            # extract out nodes based on tags, then separate into src/dst point cloud
            # neighborhoods
            molecule_nodes, surface_nodes, substrate_nodes = self._extract_indices(
                tags, nodes
            )
            src_nodes, dst_nodes = self._pick_src_dst(
                molecule_nodes, surface_nodes, substrate_nodes
            )
            # create point cloud featurizations
            src_features = atomic_numbers[src_nodes].long()
            dst_features = atomic_numbers[dst_nodes].long()
            pc_features = utils.point_cloud_featurization(
                src_features, dst_features, max_types=100
            )
            node_pos = g.ndata["pos"]
            if self.full_pairwise:
                node_pos = node_pos[dst_nodes][None, :] - node_pos[src_nodes][:, None]
            # copy data over to dictionary
            data["pc_features"] = pc_features
            data["pos"] = node_pos
            data["num_centers"] = len(src_nodes)
            data["num_neighbors"] = len(dst_nodes)
            data["force"] = g.ndata["force"][dst_nodes].squeeze()
