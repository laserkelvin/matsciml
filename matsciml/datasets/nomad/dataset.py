from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from matsciml.common.registry import registry
from matsciml.common.types import BatchDict, DataDict
from matsciml.datasets.base import PointCloudDataset
from matsciml.datasets.utils import (
    atomic_number_map,
    concatenate_keys,
    point_cloud_featurization,
)


@registry.register_dataset("NomadDataset")
class NomadDataset(PointCloudDataset):
    __devset__ = Path(__file__).parents[0].joinpath("devset")

    def index_to_key(self, index: int) -> tuple[int]:
        return (0, index)

    @staticmethod
    def collate_fn(batch: list[DataDict]) -> BatchDict:
        return concatenate_keys(
            batch,
            pad_keys=["pc_features"],
            unpacked_keys=["sizes", "src_nodes", "dst_nodes"],
        )

    def raw_sample(self, idx):
        return super().data_from_key(0, idx)

    @property
    def target_keys(self) -> dict[str, list[str]]:
        """Specifies tasks and their target keys. If more labels are desired this is
        they should be added by hand.

        Returns:
            Dict[str, List[str]]: target keys
        """
        return {
            "regression": ["relative_energy", "energy_total", "efermi"],
            "classification": ["spin_polarized"],
        }

    def target_key_list(self):
        keys = []
        for k, v in self.target_keys.items():
            keys.extend(v)
        return keys

    @staticmethod
    def _standardize_values(
        value: float | Iterable[float],
    ) -> torch.Tensor | float:
        """
        Standardizes targets to be ingested by a model.

        For scalar values, we simply return it unmodified, because they can be easily collated.
        For iterables such as tuples and NumPy arrays, we use the appropriate tensor creation
        method, and typecasted into FP32 or Long tensors.

        The type hint `float` is used to denote numeric types more broadly.

        Parameters
        ----------
        value : Union[float, Iterable[float]]
            A target value, which can be a scalar or array of values

        Returns
        -------
        Union[torch.Tensor, float]
            Mapped torch.Tensor format, or a scalar numeric value
        """
        if isinstance(value, Iterable) and not isinstance(value, str):
            # get type from first entry
            dtype = torch.long if isinstance(value[0], int) else torch.float
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value).type(dtype)
            else:
                return torch.Tensor(value).type(dtype)
        # for missing data, set to zero
        elif value is None:
            return 0.0
        else:
            # for scalars, just return the value
            return value

    def _parse_data(self, data: dict[str, Any], return_dict: dict[str, Any]) -> dict:
        """Parse out relevant data and store it in a MatSciML friendly format.

        Args:
            data (Dict[str, Any]): Data from nomad request
            return_dict (Dict[str, Any]): Empty dict to be filled with data

        Returns:
            Dict: Data compatible with MatSciML
        """
        cart_coords = (
            torch.Tensor(
                data["properties"]["structures"]["structure_original"][
                    "cartesian_site_positions"
                ],
            )
            * 1e10
        )
        system_size = len(cart_coords)
        return_dict["pos"] = cart_coords
        # getattr is used so that we can default to False; this comes
        # into play when transforms are being initialized and full_pairwise
        # as an attribute might be missing
        chosen_nodes = self.choose_dst_nodes(
            system_size, getattr(self, "full_pairwise", False)
        )
        src_nodes, dst_nodes = chosen_nodes["src_nodes"], chosen_nodes["dst_nodes"]

        atomic_numbers = torch.LongTensor(
            [
                atomic_number_map()[symbol]
                for symbol in data["properties"]["structures"]["structure_original"][
                    "species_at_sites"
                ]
            ],
        )
        return_dict["atomic_numbers"] = atomic_numbers
        return_dict["cart_coords"] = cart_coords
        # uses one-hot encoding featurization
        pc_features = point_cloud_featurization(
            atomic_numbers[src_nodes],
            atomic_numbers[dst_nodes],
            100,
        )
        # keep atomic numbers for graph featurization
        return_dict["pc_features"] = pc_features
        return_dict["sizes"] = system_size
        return_dict.update(**chosen_nodes)

        # space_group = structure.get_space_group_info()[-1]
        # # convert lattice angles into radians
        lattice_params = data["properties"]["structures"]["structure_original"][
            "lattice_parameters"
        ]
        lattice_abc = (
            lattice_params["a"] * 1e10,
            lattice_params["b"] * 1e10,
            lattice_params["c"] * 1e10,
        )
        lattice_angles = (
            lattice_params["alpha"],
            lattice_params["beta"],
            lattice_params["gamma"],
        )
        # Need to check if angles are in rad or deg
        lattice_params = torch.FloatTensor(lattice_abc + lattice_angles)
        return_dict["lattice_params"] = lattice_params
        band_structure = data["properties"]["electronic"]["band_structure_electronic"]
        if isinstance(band_structure, list):
            band_structure = band_structure[-1]  # Take the last value from the list
        return_dict["efermi"] = band_structure["energy_fermi"] * 6.241509e18
        return_dict["energy_total"] = data["energies"]["total"]["value"] * 6.241509e18
        # data['properties']['electronic']['dos_electronic']['energy_fermi']
        return_dict["spin_polarized"] = band_structure["spin_polarized"]
        return_dict["symmetry"] = {}
        return_dict["symmetry"]["number"] = data["material"]["symmetry"][
            "space_group_number"
        ]
        return_dict["symmetry"]["symbol"] = data["material"]["symmetry"][
            "space_group_symbol"
        ]
        return_dict["symmetry"]["group"] = data["material"]["symmetry"]["point_group"]
        standard_keys = set(return_dict.keys()).difference(
            ["symmetry", "spin_polarized"],
        )
        standard_dict = {
            key: self._standardize_values(return_dict[key]) for key in standard_keys
        }
        return_dict.update(standard_dict)
        target_keys = self.target_key_list()
        targets = {
            key: self._standardize_values(return_dict[key]) for key in target_keys
        }
        return_dict["targets"] = targets

        target_types = {"regression": [], "classification": []}
        for key in target_keys:
            item = targets.get(key)
            if isinstance(item, Iterable):
                # check if the data is numeric first
                if isinstance(item[0], (float, int)):
                    target_types["regression"].append(key)
            else:
                if isinstance(item, (float, int)):
                    target_type = (
                        "classification" if isinstance(item, int) else "regression"
                    )
                    target_types[target_type].append(key)

        return_dict["target_types"] = target_types
        return return_dict

    def data_from_key(self, lmdb_index: int, subindex: int) -> Any:
        # for a full list of properties avaialbe:
        # data['properties']['available_properties'
        # additional energy properties also available:
        # data['energies'].keys()
        data = super().data_from_key(lmdb_index, subindex)
        return_dict = {}
        for k in ["reference_energy", "reference_structure", "relative_energy"]:
            return_dict[k] = data[k]
        self._parse_data(data, return_dict=return_dict)
        return return_dict
