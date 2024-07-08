from __future__ import annotations


from dataclasses import field, dataclass
from typing import Any, Callable, Union

import torch
from jaxtyping import Float, Int, Real, jaxtyped
from beartype import beartype

from matsciml.common import package_registry

__all__ = [
    "ModelingTypes",
    "GraphTypes",
    "DataType",
    "AbstractGraph",
    "DataDict",
    "BatchDict",
    "Embeddings",
    "AtomicStructure",
]

# for point clouds
representations = [torch.Tensor]
graph_types = []

if package_registry["pyg"]:
    from torch_geometric.data import Data as PyGGraph

    representations.append(PyGGraph)
    graph_types.append(PyGGraph)
if package_registry["dgl"]:
    from dgl import DGLGraph

    representations.append(DGLGraph)
    graph_types.append(DGLGraph)

ModelingTypes = tuple(representations)
GraphTypes = tuple(graph_types)

DataType = Union[ModelingTypes]
AbstractGraph = Union[GraphTypes]

# for a dictionary look up of data
DataDict = dict[str, Union[float, DataType]]

# for a dictionary of batched data
BatchDict = dict[str, Union[float, DataType, DataDict]]

# for specific tensors, we define expected shapes
# we always expect at least primitive coordinates to be 2D
CoordinateTensor = Int[torch.Tensor, "nodes 3"]
# these are used for type and shape checking
# typically this represents [`num_graphs`]
ScalarTensor = Real[torch.Tensor, ""]
# typically this represents [`num_nodes`, dim] like forces
FieldTensor = Real[torch.Tensor, "a ... b"]
PointEmbeddingTensor = Float[torch.Tensor, "nodes ... dim"]
SystemEmbeddingTensor = Float[torch.Tensor, "graphs ... dim"]
# for storing edges, we choose the PyG [2, num_edges] format
EdgeTensor = Int[torch.Tensor, "2 edges"]
CellTensor = Float[torch.Tensor, "_ 3 3"]
ImageTensor = Float[torch.Tensor, "nodes 3"]


class _PydanticConfig:
    arbitrary_types_allowed = True


@jaxtyped(typechecker=beartype)
@dataclass
class Embeddings:
    """
    Data structure that packs together embeddings from a model.
    """

    system_embedding: torch.Tensor | None = None
    point_embedding: torch.Tensor | None = None
    reduction: str | Callable | None = None
    reduction_kwargs: dict[str, str | float] = field(default_factory=dict)

    @property
    def num_points(self) -> int:
        if not isinstance(self.point_embedding, torch.Tensor):
            raise ValueError("No point-level embeddings stored!")
        return self.point_embedding.size(0)

    @property
    def batch_size(self) -> int:
        if not isinstance(self.system_embedding, torch.Tensor):
            raise ValueError(
                "No system-level embeddings stored, can't determine batch size!",
            )
        return self.system_embedding.size(0)

    def reduce_point_embeddings(
        self,
        reduction: str | Callable | None = None,
        **reduction_kwargs,
    ) -> torch.Tensor:
        """
        Perform a reduction/readout of the point-level embeddings to obtain
        system/graph-level embeddings.

        This function provides a regular interface for obtaining system-level
        embeddings by either passing a function that functions via:

        ``system_level = reduce(point_level)``

        or by passing a ``str`` name of a function from ``torch`` such as ``mean``.
        """
        assert isinstance(
            self.point_embedding,
            torch.Tensor,
        ), "No point-level embeddings stored to reduce."
        if not reduction:
            reduction = self.reduction
        if isinstance(reduction, str):
            reduction = getattr(torch, reduction)
        if not reduction:
            raise ValueError("No method for reduction passed.")
        self.reduction_kwargs.update(reduction_kwargs)
        system_embeddings = reduction(self.point_embedding, **self.reduction_kwargs)
        self.system_embedding = system_embeddings
        return system_embeddings


@jaxtyped(typechecker=beartype)
@dataclass
class AtomicStructure:
    """
    Implements a data structure holding an atomic structure point cloud.

    This serves as a basis for more specialized structures by providing
    some commonly used routines and patterns.

    Attributes
    ----------
    pos : CoordinateTensor
        2D tensor comprising coordinates of atoms.
        Expected shape is [num_atoms, 3].
    atomic_numbers : ScalarTensor
        1D atomic numbers of each atom. Expected shape is [num_atoms]
    targets : dict[str, ScalarTensor | FieldTensor | float]
        Dictionary of ground truth values. Can be float, or an N-D tensor.
    target_keys : dict[str, list[str]]
        Target keys for each task category (regression/classification).
        Under each category is a list of keys that refer to the ``targets``
        dictionary.
    """

    pos: CoordinateTensor
    atomic_numbers: ScalarTensor
    targets: dict[str, ScalarTensor | FieldTensor | float]
    target_keys: dict[str, list[str]]
    dataset: str
    sample_index: int = 0
    point_group: int | None = None
    cell: CellTensor | None = None
    images: ImageTensor | None = None
    pc_features: torch.FloatTensor | None = None

    @property
    def num_atoms(self) -> int:
        """Get the number of atoms in the structure."""
        return len(self.atomic_numbers)

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        """
        Return the tensors contained within this data structure,
        including within ``targets``.

        Returns
        -------
        dict[str, torch.Tensor]
            Collection of tensors, using the original key it
            was found as.
        """
        return_dict = {}
        for key in self.__dict__:
            obj = getattr(self, key)
            if isinstance(obj, torch.Tensor):
                return_dict[key] = obj
        for key, tensor in self.targets.items():
            if isinstance(tensor, torch.Tensor):
                return_dict[key] = tensor
        return return_dict

    def __getitem__(self, key: str) -> Any:
        """
        Looks into the current structure for a specified key.

        If the key is present in either the top level, or within
        the ``targets`` dictionary, we return the value it references.

        Parameters
        ----------
        key
            Key to search within the data structure for.

        Returns
        -------
        Any
            What ever the referenced value represents.

        Raises
        ------
        KeyError:
            If the key does not exist in either the top level data
            structure or the ``targets`` dictionary, we raise a
            ``KeyError`` to inform the user of a missing key.
        """
        if key in dir(self):
            return getattr(self, key)
        if key in self.targets:
            return self.targets.get(key)
        else:
            raise KeyError(f"{key} is not an input or target of {self.dataset}")

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a reference key to a given value.

        If the key exists in ``targets``, we will override
        the existing ``targets`` entry. Otherwise, we will
        just set it as a top level attribute.

        Parameters
        ----------
        key : str
            Name of the attribute or target to write to.
        value : Any
            Object to stash in the structure.
        """
        if key in self.targets:
            self.targets[key] = value
        else:
            setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        """
        Deletes an attribute from the data structure.

        Will first check the presence of the key at the top level
        structure, and if it isn't there, check in ``targets``.

        Parameters
        ----------
        key : str
            Key to search for within the data structure.

        Raises
        ------
        KeyError:
            If the key is missing from either the top level structure
            or from the ``targets`` dictionary.
        """
        if key in dir(self):
            delattr(self, key)
        elif key in self.targets:
            del self.targets[key]
        else:
            raise KeyError(f"Attempted to delete {key}, but is not in {self.dataset}.")

    def __contains__(self, key: str) -> bool:
        """
        Implements magic method for checking if a key is contained in
        the data structure, including both top level and ``targets``.
        Notably, this does not only refer to tensors, but also all
        data types.

        Parameters
        ----------
        key : str
            Key to check for availability in the data structure.

        Returns
        -------
        bool
            True if the key is in either the structure or within ``targets``.
        """
        tensor_keys = set(self.tensors.keys())
        top_level_keys = set(self.__dict__.keys())
        all_keys = tensor_keys.union(top_level_keys)
        return key in all_keys

    def set_require_grad(self, key: str, state: bool | None = None) -> None:
        """
        Sets the ``requires_grad`` state of a tensor within the structure.

        If a ``state`` is passed, it will be set to that state explicitly.
        Otherwise, the default behavior will flip the existing state, i.e.
        if ``requires_grad`` for a tensor was originally ``False``, calling
        this function on a tensor referenced by ``key`` will set it to ``True``.

        Parameters
        ----------
        key : str
            Key referencing the tensor, which is expected to be
            contained in the ``tensors`` class property.
        state : bool | None, default None
            Explicit state to set the ``requires_grad`` to. By
            default None, which will negate the existing state.

        Raises
        ------
        KeyError:
            If the tensor cannot be found within this data structure.
        """
        if key not in self.tensors.keys():
            raise KeyError(
                f"{key} was specified for gradients, but absent from {self.dataset}."
            )
        if not isinstance(self.tensors[key], torch.FloatTensor):
            raise RuntimeError(
                f"Tensor {key} in {self.dataset} was required to have gradients, but is not a float tensor."
            )
        # if no state is provided, we toggle the flag (i.e. enable gradients if it's off)
        if state is None:
            state = not self.tensors[key].requires_grad
        self.tensors[key].requires_grad_(state)

    def to(
        self, device: str | torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        Perform an in-place (as far as Python is concerned) operation to move
        or typecast tensors within the data structure.

        This provides a similar interface to PyTorch for collectively
        performing operations on a set of tensors.

        Parameters
        ----------
        device : str | torch.device | None, default None
            Device to move all tensors to, by default None which
            performs no movement (i.e. to itself).
        dtype : torch.dtype | None, default None
            Data type to cast all tensors to, by default None
            which does not perform any casting.
        """
        for key, tensor in self.tensors.items():
            if dtype is None:
                dtype = tensor.dtype
            # figure out where the tensor came from in the data structure
            if key in self.__dict__:
                setattr(self, key, tensor.to(device, dtype))
            # assume it came from targets dict
            else:
                self.targets[key] = tensor.to(device, dtype)
        # target keys are nested under regression/classification
        for group in self.target_keys.values():
            for target_name in group:
                value = self.targets[target_name]
                if isinstance(value, torch.Tensor):
                    self.targets[target_name] = value.to(device, dtype)


class GraphStructure(AtomicStructure):
    _graph: AbstractGraph
    offsets: ImageTensor
    graph_keys: list[str] | None = None

    @property
    def is_pyg(self) -> bool:
        """
        Simple property to report if the graph is PyG or not.

        This should readily simplify routines that need to determine
        how to act on/with a graph.

        Returns
        -------
        bool
            True if the graph is from ``torch_geometric``, False
            if it isn't or if PyG is not installed.
        """
        if package_registry["pyg"]:
            return isinstance(self._graph, PyGGraph)
        return False

    @property
    def edges(self) -> EdgeTensor:
        """
        Return the edges of the graph.

        For PyG we return the edge index as is, and for DGLGraph
        we stack the ``src``/``dst`` nodes together to match what
        is expected for ``EdgeTensor``.

        Returns
        -------
        EdgeTensor
            Long tensor of shape [2, num_edges].
        """
        if self.is_pyg:
            return self._graph.edge_index
        else:
            src, dst = self._graph.edges
            return torch.stack([src, dst])

    @property
    def src_nodes(self) -> torch.LongTensor:
        return self.edges[0, :]

    @property
    def dst_nodes(self) -> torch.LongTensor:
        return self.edges[1, :]

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        tensor_dict = super().tensors
        # for DGL graphs, we have to look in ndata/edata
        if not self.is_pyg:
            for n_key, ndata in self._graph.ndata.items():
                tensor_dict[n_key] = ndata
            for e_key, edata in self._graph.edata.items():
                tensor_dict[e_key] = edata
        return tensor_dict

    @property
    def num_nodes(self) -> int:
        return self.num_atoms

    @property
    def num_edges(self) -> int:
        return self.edges.size(-1)

    @property
    def node_data(self) -> dict[str, torch.Tensor]:
        """
        Return tensors that represent node features.

        This property naively assumes that if the first dimension
        of any tensor matches the number of nodes, it is a node
        feature.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping of attribute name and tensor
            that corresponds to a node feature.
        """
        data = {}
        for key, tensor in self.tensors.items():
            if tensor.size(0) == self.num_nodes:
                data[key] = tensor
        return data

    @property
    def edge_data(self) -> dict[str, torch.Tensor]:
        """
        Return tensors that represent edge features.

        This property naively assumes that if the first dimension
        of any tensor matches the number of edges, it is an edge
        feature.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping of attribute name and tensor
            that corresponds to an edge feature.
        """
        data = {}
        for key, tensor in self.tensors.items():
            if tensor.size(0) == self.num_edges:
                data[key] = tensor
        return data

    @property
    def graph_data(self) -> dict[str, ScalarTensor | float]:
        """
        This property returns data referenced by ``graph_keys``.

        This requires the user/developer to define what properties
        are considered graph-level variables/features as there
        is no real way to determine what they are by shape inspection.

        Returns
        -------
        dict[str, ScalarTensor | float]
            Dictionary mapping of key/tensor for graph variables,
            according to ``graph_keys``.

        Raises
        ------
        RuntimeError:
            If there are no ``graph_keys`` set at creation, this
            property cannot work and we raise a ``RuntimeError``.
        KeyError:
            If a key in ``graph_keys`` is not found in the data
            structure, we raise a ``KeyError``.
        """
        if self.graph_keys is None:
            raise RuntimeError(
                "Expected to use graph-level features, but ``graph_keys`` was not set."
            )
        graph_feats = {}
        for key in self.graph_keys:
            if key not in self:
                raise KeyError(
                    f"Expected {key} to be in {self.dataset} but was not found."
                )
            graph_feats[key] = self[key]
        return graph_feats

    def local_scope(self):
        """
        Wraps the graph local scope context for DGL.

        Raises
        ------
        RuntimeError:
            If the graph is a PyG graph, which does not have
            this context manager implemented, we raise a
            ``RuntimeError``.
        """
        if self.is_pyg:
            raise RuntimeError("PyG does not have local scope implemented.")
        return self._graph.local_scope


class BatchMixin:
    @property
    def batch_size(self) -> int:
        """Returns the number of samples in the batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self._batch_size = value

    @property
    def batch(self) -> torch.LongTensor:
        """
        Returns a ``batch`` tensor in the PyG sense.

        This provides a ``LongTensor`` with shape ``[num_nodes]``,
        where the value of each element corresponds to the sample
        index each node belongs to, similar to how PyG indicates
        membership.

        Returns
        -------
        torch.LongTensor
            LongTensor containing sample indices for each node.
        """
        return self._batch

    @batch.setter
    def batch(self, values: list[int] | torch.Tensor) -> None:
        if isinstance(values, list):
            values = torch.LongTensor(values)
        if not values.dtype == torch.long:
            values = values.long()
        self._batch = values

    @property
    def mask(self) -> torch.Tensor | None:
        """Returns a mask for point clouds."""
        return self._mask

    @mask.setter
    def mask(self, values: torch.Tensor | None) -> None:
        self._mask = values
