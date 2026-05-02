import pytest

import mtlearn

if not getattr(mtlearn, "WITH_TORCH", False):
    pytest.skip("build has no LibTorch support", allow_module_level=True)

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Python dependency unavailable: {exc}", allow_module_level=True)

from mtlearn import morphology
from mtlearn.layers import (
    ConnectedFilterPreprocessingLayer,
    ConnectedFilterPreprocessingLayerWithCPUTreeTraversal,
    ConnectedFilterPreprocessingLayerWithExplicitJacobian,
)

pytestmark = pytest.mark.integration


def _tiny_dataset_loader(batch_size=2):
    x = torch.tensor(
        [
            [[[2, 2, 0], [2, 5, 0], [3, 3, 1]]],
            [[[1, 0, 1], [4, 4, 2], [0, 2, 2]]],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 1], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _single_area_layer(layer_cls=ConnectedFilterPreprocessingLayer, *, scale_mode="minmax01"):
    return layer_cls(
        in_channels=1,
        attributes_spec=[(morphology.AttributeType.AREA,)],
        tree_type="max-tree",
        device="cpu",
        scale_mode=scale_mode,
        beta_f=1.0,
        clamp_logits=False,
    )


def test_build_dataloader_cached_populates_primary_layer_cache():
    layer = _single_area_layer(scale_mode="minmax01")
    loader = _tiny_dataset_loader()

    cached_loader = layer.build_dataloader_cached(loader)

    assert layer._stats_frozen is True
    assert set(layer._info_jacobian) == {"0_0", "1_0"}
    assert set(layer._base_attrs) == {"0_0", "1_0"}
    assert set(layer._norm_attrs) == {"0_0", "1_0"}
    assert layer._stats_epoch > 0
    assert all(epoch == layer._stats_epoch for epoch in layer._norm_epoch_by_key.values())

    ((x, idx), y) = next(iter(cached_loader))
    assert x.shape == (2, 1, 3, 3)
    assert idx.tolist() == [0, 1]
    assert y.tolist() == [0, 1]


def test_cached_forward_matches_uncached_forward_with_same_parameters():
    x, _ = next(iter(_tiny_dataset_loader()))
    cached_layer = _single_area_layer(scale_mode="none")
    plain_layer = _single_area_layer(scale_mode="none")
    plain_layer.load_state_dict(cached_layer.state_dict())

    cached_loader = cached_layer.build_dataloader_cached(_tiny_dataset_loader())
    cached_input, _ = next(iter(cached_loader))

    y_cached = cached_layer(cached_input)
    y_plain = plain_layer(x)

    assert y_cached.shape == y_plain.shape
    assert torch.allclose(y_cached, y_plain)


def test_freeze_and_unfreeze_dataset_stats_controls_stat_updates():
    layer = _single_area_layer(scale_mode="minmax01")
    attr = morphology.AttributeType.AREA

    layer._update_ds_stats(attr, torch.tensor([2.0, 4.0]))
    initial_epoch = layer._stats_epoch
    initial_min = layer._ds_stats[attr]["amin"].clone()

    layer.freeze_ds_stats()
    layer._update_ds_stats(attr, torch.tensor([0.0, 10.0]))

    assert layer._stats_epoch == initial_epoch
    assert torch.equal(layer._ds_stats[attr]["amin"], initial_min)

    layer.unfreeze_ds_stats()
    layer._update_ds_stats(attr, torch.tensor([0.0, 10.0]))

    assert layer._stats_epoch == initial_epoch + 1
    assert layer._ds_stats[attr]["amin"].item() == 0.0


def test_refresh_cached_normalization_uses_latest_stats():
    layer = _single_area_layer(scale_mode="minmax01")
    layer.build_dataloader_cached(_tiny_dataset_loader())

    attr = morphology.AttributeType.AREA
    before = layer._norm_attrs["0_0"][attr].clone()

    layer._ds_stats[attr]["amin"] = torch.tensor(0.0)
    layer._ds_stats[attr]["amax"] = torch.tensor(100.0)
    layer._stats_epoch += 1
    layer.refresh_cached_normalization()

    after = layer._norm_attrs["0_0"][attr]
    assert not torch.allclose(after, before)
    assert layer._norm_epoch_by_key["0_0"] == layer._stats_epoch


@pytest.mark.parametrize(
    "layer_cls",
    [
        ConnectedFilterPreprocessingLayer,
        ConnectedFilterPreprocessingLayerWithExplicitJacobian,
        ConnectedFilterPreprocessingLayerWithCPUTreeTraversal,
    ],
)
def test_load_stats_roundtrip_after_save(tmp_path, layer_cls):
    source = _single_area_layer(layer_cls, scale_mode="minmax01")
    source.build_dataloader_cached(_tiny_dataset_loader())
    stats_path = tmp_path / "stats.pt"

    source.save_stats(str(stats_path))

    target = _single_area_layer(layer_cls, scale_mode="minmax01")
    target.load_stats(str(stats_path))

    assert target._ds_stats.keys() == source._ds_stats.keys()
    attr = morphology.AttributeType.AREA
    assert torch.equal(target._ds_stats[attr]["amin"], source._ds_stats[attr]["amin"])
    assert torch.equal(target._ds_stats[attr]["amax"], source._ds_stats[attr]["amax"])
