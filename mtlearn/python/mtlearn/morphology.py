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


def create_max_tree(image):
    return _backend.create_max_tree(image)


def create_min_tree(image):
    return _backend.create_min_tree(image)


def create_tree_of_shapes(image):
    return _backend.create_tree_of_shapes(image)


def build_tree(image, tree_type: str):
    return _backend.build_tree(image, tree_type)


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
    "create_max_tree",
    "create_min_tree",
    "create_tree_of_shapes",
    "build_tree",
    "compute_attributes",
    "compute_single_attribute",
    "describe_attribute",
    "describe_all_attributes",
    "create_attribute_filter",
    "is_tree",
]
