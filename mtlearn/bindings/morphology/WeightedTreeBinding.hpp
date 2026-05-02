#pragma once

// Pybind exposure for the mtlearn WeightedTree facade.
//
// This file intentionally exposes a rich query surface because existing
// notebooks inspect component-tree topology directly. The C++ public facade
// remains smaller; topology-heavy methods are routed through internal backend
// accessors here so Python can keep its current behavior without making those
// backend details part of the installed C++ API.

#include "BindingSupport.hpp"

#include <optional>
#include <vector>

namespace mtlearn {
namespace morphology_pybind {

// Convert backend traversal ranges into Python-friendly vectors. Backend
// iterators are often lightweight views, so bindings materialize them before
// returning to Python.
template <class Range>
std::vector<morphology::NodeId> collectNodeIds(const Range& range)
{
    std::vector<morphology::NodeId> ids;
    for (morphology::NodeId id : range) {
        ids.push_back(id);
    }
    return ids;
}

// A connected component is represented by a node plus all proper parts in its
// subtree. This helper is used by reconstructNode to produce a binary mask for
// inspection/debugging in notebooks.
inline std::vector<int> collectPixelsOfConnectedComponent(
    const morphology::detail::TreeTopology& tree,
    morphology::NodeId nodeId)
{
    std::vector<int> pixels;
    for (morphology::NodeId subtreeNodeId : tree.getNodeSubtree(nodeId)) {
        for (int properPart : tree.getProperParts(subtreeNodeId)) {
            pixels.push_back(properPart);
        }
    }
    return pixels;
}

// Build a uint8 mask for one connected component. The output is not the tree
// reconstruction image; it is an inspection aid that marks pixels covered by a
// selected node.
inline py::array_t<uint8_t> reconstructNode(const morphology::detail::TreeTopology& tree, morphology::NodeId nodeId)
{
    if (!tree.isNode(nodeId) || !tree.isAlive(nodeId)) {
        throw std::invalid_argument("invalid NodeId for reconstruction");
    }

    auto image = mmcfilters::ImageUInt8::create(tree.getNumRowsOfImage(), tree.getNumColsOfImage());
    image->fill(0);
    for (int pixel : collectPixelsOfConnectedComponent(tree, nodeId)) {
        (*image)[pixel] = 255;
    }
    return imageToNumpy(image);
}

// Register shared morphology enums at module level. Attribute-specific enums
// are nested under Attribute in AttributeBinding.hpp.
inline void bindCoreMorphologyEnums(py::module& m)
{
    py::enum_<morphology::TreeOfShapesInterpolation>(m, "ToSInterpolation", py::module_local())
        .value("SelfDual", morphology::TreeOfShapesInterpolation::SelfDual)
        .value("Min4cMax8c", morphology::TreeOfShapesInterpolation::Min4cMax8c)
        .value("Min8cMax4c", morphology::TreeOfShapesInterpolation::Min8cMax4c)
        .export_values();

    py::enum_<morphology::NodeIdSpace>(m, "NodeIdSpace", py::module_local())
        .value("MORPHOLOGICAL_TREE", morphology::NodeIdSpace::MORPHOLOGICAL_TREE)
        .value("HIGRA", morphology::NodeIdSpace::HIGRA)
        .export_values();
}

// Attach topology, traversal, and mutation methods to the Python
// WeightedMorphologicalTree class. Multiple naming styles are intentionally
// preserved because notebooks historically used camelCase and snake_case.
template <class PyClass>
void bindWeightedTreeQueries(PyClass& cls)
{
    cls.def_property_readonly("numInternalNodeSlots", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getNumInternalNodeSlots();
        })
        .def_property_readonly("numTotalProperParts", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getNumTotalProperParts();
        })
        .def_property_readonly("numHigraNodes", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getNumHigraNodes();
        })
        .def("getRoot", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getRoot();
        })
        .def_property_readonly("root", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getRoot();
        })
        .def_property_readonly("numFreeNodeSlots", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getNumFreeNodeSlots();
        })
        .def_property_readonly("numLeafNodes", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getNumLeafNodes();
        })
        .def("getAliveNodeIds", [](morphology::WeightedTree& self) {
            return collectNodeIds(morphology::detail::topology(self).getAliveNodeIds());
        })
        .def_property_readonly("aliveNodeIds", [](morphology::WeightedTree& self) {
            return collectNodeIds(morphology::detail::topology(self).getAliveNodeIds());
        })
        .def_property_readonly("alive_node_ids", [](morphology::WeightedTree& self) {
            return collectNodeIds(morphology::detail::topology(self).getAliveNodeIds());
        })
        .def("getLeafNodeIds", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getLeaves();
        })
        .def_property_readonly("leafNodeIds", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getLeaves();
        })
        .def_property_readonly("leaf_node_ids", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getLeaves();
        })
        .def("getChildren", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).getChildren(nodeId));
        }, "nodeId"_a)
        .def("childrenOf", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).getChildren(nodeId));
        }, "nodeId"_a)
        .def("children_of", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).getChildren(nodeId));
        }, "nodeId"_a)
        .def("getNodeNumDescendants", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getNodeNumDescendants(nodeId);
        }, "nodeId"_a)
        .def("getNodeNumSiblings", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getNodeNumSiblings(nodeId);
        }, "nodeId"_a)
        .def("getNumProperParts", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getNumProperParts(nodeId);
        }, "nodeId"_a)
        .def("getNodeTimePreOrder", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getNodeTimePreOrder(nodeId);
        }, "nodeId"_a)
        .def("getNodeTimePostOrder", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getNodeTimePostOrder(nodeId);
        }, "nodeId"_a)
        .def("getProperParts", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).getProperParts(nodeId));
        }, "nodeId"_a)
        .def("properPartsOf", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).getProperParts(nodeId));
        }, "nodeId"_a)
        .def("proper_parts_of", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).getProperParts(nodeId));
        }, "nodeId"_a)
        .def("reconstructNode", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return reconstructNode(morphology::detail::topology(self), nodeId);
        }, "nodeId"_a)
        .def("getPostOrderNodes", [](morphology::WeightedTree& self, std::optional<morphology::NodeId> rootNodeId) {
            return rootNodeId.has_value()
                ? collectNodeIds(morphology::detail::topology(self).getPostOrderNodes(*rootNodeId))
                : collectNodeIds(morphology::detail::topology(self).getPostOrderNodes());
        }, "rootNodeId"_a = std::nullopt)
        .def("getIteratorBreadthFirstTraversal", [](morphology::WeightedTree& self, std::optional<morphology::NodeId> rootNodeId) {
            return rootNodeId.has_value()
                ? collectNodeIds(morphology::detail::topology(self).getIteratorBreadthFirstTraversal(*rootNodeId))
                : collectNodeIds(morphology::detail::topology(self).getIteratorBreadthFirstTraversal());
        }, "rootNodeId"_a = std::nullopt)
        .def("getPathToRootNodes", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).getPathToRootNodes(nodeId));
        }, "nodeId"_a)
        .def("getPathBetweenNodes", [](morphology::WeightedTree& self, morphology::NodeId sourceNodeId, morphology::NodeId targetNodeId) {
            return collectNodeIds(morphology::detail::topology(self).getPathBetweenNodes(sourceNodeId, targetNodeId));
        }, "sourceNodeId"_a, "targetNodeId"_a)
        .def("getNodeSubtree", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).getNodeSubtree(nodeId));
        }, "nodeId"_a)
        .def("nodeSubtreeOf", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).getNodeSubtree(nodeId));
        }, "nodeId"_a)
        .def("node_subtree_of", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).getNodeSubtree(nodeId));
        }, "nodeId"_a)
        .def("getDescendants", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).getDescendants(nodeId));
        }, "nodeId"_a)
        .def("descendantsOf", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).getDescendants(nodeId));
        }, "nodeId"_a)
        .def("descendants_of", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).getDescendants(nodeId));
        }, "nodeId"_a)
        .def("getNodeParent", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getNodeParent(nodeId);
        }, "nodeId"_a)
        .def("parentOf", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getNodeParent(nodeId);
        }, "nodeId"_a)
        .def("parent_of", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getNodeParent(nodeId);
        }, "nodeId"_a)
        .def("getSmallestComponent", [](morphology::WeightedTree& self, int pixelId) {
            return morphology::detail::topology(self).getSmallestComponent(pixelId);
        }, "pixelId"_a)
        .def("smallestComponentOf", [](morphology::WeightedTree& self, int pixelId) {
            return morphology::detail::topology(self).getSmallestComponent(pixelId);
        }, "pixelId"_a)
        .def("smallest_component_of", [](morphology::WeightedTree& self, int pixelId) {
            return morphology::detail::topology(self).getSmallestComponent(pixelId);
        }, "pixelId"_a)
        .def("getHigraNodeId", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getHigraNodeId(nodeId);
        }, "nodeId"_a)
        .def("getNumChildren", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getNumChildren(nodeId);
        }, "nodeId"_a)
        .def("getFirstChild", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getFirstChild(nodeId);
        }, "nodeId"_a)
        .def("getNextSibling", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getNextSibling(nodeId);
        }, "nodeId"_a)
        .def("isNode", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).isNode(nodeId);
        }, "nodeId"_a)
        .def("isProperPart", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).isProperPart(nodeId);
        }, "nodeId"_a)
        .def("isAlive", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).isAlive(nodeId);
        }, "nodeId"_a)
        .def("isRoot", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).isRoot(nodeId);
        }, "nodeId"_a)
        .def("isLeaf", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).isLeaf(nodeId);
        }, "nodeId"_a)
        .def("hasChild", [](morphology::WeightedTree& self, morphology::NodeId parentId, morphology::NodeId childId) {
            return morphology::detail::topology(self).hasChild(parentId, childId);
        }, "parentId"_a, "childId"_a)
        .def("pruneNode", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            self.pruneNode(nodeId);
        }, "nodeId"_a)
        .def("mergeNodeIntoParent", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            self.mergeNodeIntoParent(nodeId);
        }, "nodeId"_a)
        .def_property_readonly("treeType", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getTreeType();
        })
        .def_property_readonly("hasAdjacencyRelation", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).hasAdjacencyRelation();
        })
        .def_property_readonly("hasTreeOfShapesAdjacencyPolicy", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).hasTreeOfShapesAdjacencyPolicy();
        })
        .def("getTreeOfShapesMinTreeAdjacencyRadius", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getTreeOfShapesMinTreeAdjacencyRadius();
        })
        .def("getTreeOfShapesMaxTreeAdjacencyRadius", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getTreeOfShapesMaxTreeAdjacencyRadius();
        })
        .def_property_readonly("numRows", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getNumRowsOfImage();
        })
        .def_property_readonly("numCols", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getNumColsOfImage();
        })
        .def_property_readonly("numNodes", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getNumNodes();
        })
        .def("getAltitude", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return self.getAltitude(nodeId);
        }, "nodeId"_a)
        .def("altitudeOf", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return self.getAltitude(nodeId);
        }, "nodeId"_a)
        .def("altitude_of", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return self.getAltitude(nodeId);
        }, "nodeId"_a)
        .def("getNodeResidue", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return self.getNodeResidue(nodeId);
        }, "nodeId"_a)
        .def("residueOf", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return self.getNodeResidue(nodeId);
        }, "nodeId"_a)
        .def("residue_of", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return self.getNodeResidue(nodeId);
        }, "nodeId"_a)
        .def("reconstructionImage", [](morphology::WeightedTree& self) {
            return imageToNumpy(self.reconstructionImage());
        })
        .def("exportHigraHierarchy", [](morphology::WeightedTree& self) {
            return self.exportHigraHierarchy();
        });
}

// Register constructors for max-tree, min-tree, and tree of shapes. The Python
// class stores shared_ptr<WeightedTree> so CFP tensors, filters, and attributes
// can safely share the same tree handle.
inline void bindWeightedTree(py::module& m)
{
    auto weightedTree = py::class_<morphology::WeightedTree, morphology::WeightedTreePtr>(
        m,
        "WeightedMorphologicalTree",
        py::module_local(),
        "Internal mtlearn morphology tree backed by the current morphology backend.");

    weightedTree
        .def_static("createComponentTree", [](const UInt8InputArray& input, bool isMaxTree, double radius) {
            return std::make_shared<morphology::WeightedTree>(
                morphology::WeightedTree::createComponentTree(imageViewFromArray(input), isMaxTree, radius));
        }, "input"_a, "isMaxtree"_a, "radius"_a = 1.5)
        .def_static("createMaxTree", [](const UInt8InputArray& input, double radius) {
            return std::make_shared<morphology::WeightedTree>(
                morphology::WeightedTree::createComponentTree(imageViewFromArray(input), true, radius));
        }, "input"_a, "radius"_a = 1.5)
        .def_static("createMinTree", [](const UInt8InputArray& input, double radius) {
            return std::make_shared<morphology::WeightedTree>(
                morphology::WeightedTree::createComponentTree(imageViewFromArray(input), false, radius));
        }, "input"_a, "radius"_a = 1.5)
        .def_static("createTreeOfShapes", [](const UInt8InputArray& input, morphology::TreeOfShapesInterpolation interpolation, int infinitySeedRow, int infinitySeedCol) {
            return std::make_shared<morphology::WeightedTree>(
                morphology::WeightedTree::createTreeOfShapes(imageViewFromArray(input), interpolation, infinitySeedRow, infinitySeedCol));
        },
            "input"_a,
            "interpolation"_a = morphology::TreeOfShapesInterpolation::SelfDual,
            "infinitySeedRow"_a = morphology::TreeOfShapesDefaultInfinityRow,
            "infinitySeedCol"_a = morphology::TreeOfShapesDefaultInfinityCol);

    bindWeightedTreeQueries(weightedTree);
}

} // namespace morphology_pybind
} // namespace mtlearn
