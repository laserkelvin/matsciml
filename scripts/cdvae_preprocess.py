from matsciml.datasets import MaterialsProjectDataset
from matsciml.datasets.transforms import (
    PointCloudToGraphTransform,
    MakeJImages,
    MakePyMatGenStructure,
)


dset = MaterialsProjectDataset.from_devset(
    transforms=[
        MakePyMatGenStructure(),
        MakeJImages(),
        PointCloudToGraphTransform("pyg"),
    ]
)

dset.save_preprocessed_data("cdvae_mp", 4)
