import pytest

from matsciml.datasets.transforms.labels.base import AbstractLabelTransform
from matsciml.common.registry import registry


# get datasets that will fit this
ignore_dset = ["Multi", "M3G", "PyG", "Cdvae"]
filtered_list = list(
    filter(
        lambda x: all([target_str not in x for target_str in ignore_dset]),
        registry.__entries__["datasets"].keys(),
    ),
)
label_map = {
    "MaterialsProjectDataset": "band_gap",
    "IS2REDataset": "energy",
    "S2EFDataset": "energy",
    "LiPSDataset": "energy",
    "OQMDDataset": "stability",
    "ColabFitDataset": "stress",
}
combos = [(key, value) for key, value in label_map.items()]


@pytest.mark.parametrize(
    "dset_cls_name, label_key",
    combos,
)
def test_transform_workflow(dset_cls_name: str, label_key: str):
    dset_class = registry.get_dataset_class(dset_cls_name)
    dset = dset_class.from_devset(
        transforms=[AbstractLabelTransform(label_key, agg_method="static")]
    )
    assert dset.data_sample_hash
    # try grab a few samples
    for index in range(5):
        _ = dset.__getitem__(index)
    assert dset.transforms[0].serializable_format


@pytest.mark.parametrize(
    "dset_cls_name, label_key",
    combos,
)
@pytest.mark.parametrize("agg_method", ["static", "sampled", "moving"])
def test_transform_save_cache(dset_cls_name: str, label_key: str, agg_method: str):
    dset_class = registry.get_dataset_class(dset_cls_name)
    _ = dset_class.from_devset(
        transforms=[
            AbstractLabelTransform(
                label_key, value=1.254, agg_method=agg_method, auto_save_cache=True
            )
        ]
    )
