"""Stable morphology facade for mtlearn.

The current backend is mtlearn's native ``_mtlearn`` extension. Keeping user
code behind this module lets mtlearn swap morphology backends later without
changing notebooks and high-level Python APIs.
"""

from __future__ import annotations

from typing import Iterable

from ._backends import _mtlearn as _backend


Tree = _backend.Tree
WeightedTree = _backend.Tree
WeightedMorphologicalTree = _backend.WeightedMorphologicalTree
Attribute = _backend.Attribute
AttributeType = _backend.AttributeType
AttributeGroup = _backend.AttributeGroup
AttributeFilters = _backend.AttributeFilters
NodeIdSpace = _backend.NodeIdSpace
ToSInterpolation = _backend.ToSInterpolation


def create_max_tree(image):
    return _backend.create_max_tree(image)


def create_min_tree(image):
    return _backend.create_min_tree(image)


def create_tree_of_shapes(
    image,
    interpolation=None,
    infinity_seed_row: int = 0,
    infinity_seed_col: int = 0,
):
    return _backend.create_tree_of_shapes(
        image,
        interpolation=interpolation,
        infinity_seed_row=infinity_seed_row,
        infinity_seed_col=infinity_seed_col,
    )


def build_tree(
    image,
    tree_type: str,
    *,
    tos_interpolation=None,
    tos_infinity_seed_row: int = 0,
    tos_infinity_seed_col: int = 0,
):
    return _backend.build_tree(
        image,
        tree_type,
        tos_interpolation=tos_interpolation,
        tos_infinity_seed_row=tos_infinity_seed_row,
        tos_infinity_seed_col=tos_infinity_seed_col,
    )


def normalize_tree_type(tree_type: str) -> str:
    return _backend.normalize_tree_type(tree_type)


def normalize_tos_interpolation(interpolation=None):
    return _backend.normalize_tos_interpolation(interpolation)


def compute_attributes(tree, attributes: Iterable):
    return _backend.compute_attributes(tree, attributes)


def compute_single_attribute(tree, attribute):
    return _backend.compute_single_attribute(tree, attribute)


def describe_attribute(attribute):
    return _backend.describe_attribute(attribute)


def describe_all_attributes():
    return _backend.describe_all_attributes()


def create_attribute_filter(tree):
    return _backend.create_attribute_filter(tree)


def is_tree(value) -> bool:
    return _backend.is_tree(value)


__all__ = [
    "Tree",
    "WeightedTree",
    "WeightedMorphologicalTree",
    "Attribute",
    "AttributeType",
    "AttributeGroup",
    "AttributeFilters",
    "NodeIdSpace",
    "ToSInterpolation",
    "create_max_tree",
    "create_min_tree",
    "create_tree_of_shapes",
    "build_tree",
    "normalize_tree_type",
    "normalize_tos_interpolation",
    "compute_attributes",
    "compute_single_attribute",
    "describe_attribute",
    "describe_all_attributes",
    "create_attribute_filter",
    "is_tree",
]
