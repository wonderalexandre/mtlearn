"""Adapter from the public mtlearn morphology facade to mtlearn's native bindings."""

from __future__ import annotations

from typing import Iterable

from .._native import load_bindings


_bindings = load_bindings()


Tree = _bindings.WeightedMorphologicalTree
WeightedMorphologicalTree = _bindings.WeightedMorphologicalTree
Attribute = _bindings.Attribute
AttributeType = _bindings.Attribute.Type
AttributeGroup = _bindings.Attribute.Group
AttributeFilters = _bindings.AttributeFilters
NodeIdSpace = _bindings.NodeIdSpace


def create_max_tree(image):
    return WeightedMorphologicalTree.createMaxTree(image)


def create_min_tree(image):
    return WeightedMorphologicalTree.createMinTree(image)


def create_tree_of_shapes(image):
    return WeightedMorphologicalTree.createTreeOfShapes(image)


def build_tree(image, tree_type: str):
    if tree_type == "max-tree":
        return create_max_tree(image)
    if tree_type == "min-tree":
        return create_min_tree(image)
    return create_tree_of_shapes(image)


def compute_attributes(tree, attributes: Iterable):
    return Attribute.computeAttributes(tree, attributes)


def compute_single_attribute(tree, attribute):
    return Attribute.computeSingleAttribute(tree, attribute)


def describe_attribute(attribute):
    return Attribute.describe(attribute)


def describe_all_attributes():
    return Attribute.describeAll()


def create_attribute_filter(tree):
    return AttributeFilters(tree)


def is_tree(value) -> bool:
    return isinstance(value, Tree)
