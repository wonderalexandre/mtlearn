import pytest

import mtlearn

if not getattr(mtlearn, "WITH_TORCH", False):
    pytest.skip("build has no LibTorch support", allow_module_level=True)

try:
    import torch
except Exception as exc:  # pragma: no cover
    pytest.skip(f"PyTorch unavailable: {exc}", allow_module_level=True)

from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer
from mtlearn.layers._helpers import IndexedDatasetWrapper, deserialize_ds_stats

pytestmark = pytest.mark.integration


def _single_area_layer():
    return ConnectedFilterPreprocessingLayer(
        in_channels=1,
        attributes_spec=[(morphology.AttributeType.AREA,)],
        tree_type="max-tree",
        device="cpu",
        scale_mode="none",
    )


def test_constructor_rejects_empty_attribute_group():
    with pytest.raises(ValueError, match="at least one attribute"):
        ConnectedFilterPreprocessingLayer(
            in_channels=1,
            attributes_spec=[()],
            tree_type="max-tree",
            device="cpu",
        )


@pytest.mark.parametrize(
    "attribute",
    [
        morphology.AttributeType.MAX_DIST,
        morphology.AttributeType.BITQUADS_AREA,
        morphology.AttributeGroup.BITQUADS,
    ],
)
def test_tree_of_shapes_constructor_rejects_adjacency_bound_attributes(attribute):
    with pytest.raises(ValueError, match="tree-of-shapes CFP does not support"):
        ConnectedFilterPreprocessingLayer(
            in_channels=1,
            attributes_spec=[(attribute,)],
            tree_type="tree-of-shapes",
            device="cpu",
        )


def test_forward_rejects_non_batched_input_shape():
    layer = _single_area_layer()

    with pytest.raises(AssertionError, match="expected"):
        layer(torch.zeros((1, 3, 3), dtype=torch.float32))


def test_forward_rejects_wrong_channel_count():
    layer = _single_area_layer()

    with pytest.raises(AssertionError, match="in_channels=1"):
        layer(torch.zeros((1, 2, 3, 3), dtype=torch.float32))


def test_inspection_rejects_invalid_image_rank():
    layer = _single_area_layer()

    with pytest.raises(ValueError, match="img must be"):
        layer.inspect_training_sample(torch.zeros((1, 1, 3, 3), dtype=torch.float32))


def test_inspection_missing_cache_key_requires_build_if_missing():
    layer = _single_area_layer()
    image = torch.zeros((1, 3, 3), dtype=torch.float32)

    with pytest.raises(KeyError, match="Tree/attributes not found"):
        layer.inspect_training_sample(image, idx=42, build_if_missing=False)


def test_indexed_dataset_wrapper_rejects_scalar_samples():
    class ScalarDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return torch.tensor(float(idx))

    wrapper = IndexedDatasetWrapper(ScalarDataset())

    with pytest.raises(ValueError, match="Dataset samples must be"):
        wrapper[0]


def test_deserialize_stats_rejects_unknown_attribute_key():
    with pytest.raises(ValueError, match="unknown serialized attribute key"):
        deserialize_ds_stats({"NOT_A_PUBLIC_ATTRIBUTE": {}}, torch.device("cpu"))
