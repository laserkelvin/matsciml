# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

try:
    import dgl

    _has_dgl = True
except ImportError:
    _has_dgl = False


if _has_dgl:
    from matsciml.models.dgl.dpp import DimeNetPP
    from matsciml.models.dgl.egnn import PLEGNNBackbone
    from matsciml.models.dgl.gaanet import GalaPotential
    from matsciml.models.dgl.gcn import GraphConvModel
    from matsciml.models.dgl.megnet import MEGNet
    from matsciml.models.dgl.schnet_dgl import SchNet
    from matsciml.models.dgl.mpnn import MPNN
