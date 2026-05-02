#pragma once

// Public C++ morphology facade for mtlearn.
//
// This header is the only morphology header installed as part of the public
// C++ API. It intentionally exposes mtlearn-owned types and enums instead of
// backend types from mmcfilters. The implementation may continue to use
// mmcfilters internally, but downstream C++ consumers should only need this
// facade to create trees, inspect topology, reconstruct images, and request
// attribute identifiers compatible with the Python layer.

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

namespace mtlearn::morphology {

// mtlearn uses the same integer node-id domain as the current morphology
// backend. A node id may refer either to a live tree node or to an internal
// backend slot, depending on the method being called.
using NodeId = int;
inline constexpr NodeId InvalidNode = -1;

namespace detail {
struct BackendAccess;
} // namespace detail

inline constexpr int TreeOfShapesDefaultInfinityRow = 0;
inline constexpr int TreeOfShapesDefaultInfinityCol = 0;

// Interpolation policy used when constructing a tree of shapes. The names
// match the backend concepts, but the enum is owned by mtlearn so the backend
// can be replaced without changing the public C++/Python API.
enum class TreeOfShapesInterpolation {
    SelfDual,
    Min4cMax8c,
    Min8cMax4c,
};

// Attribute computation can return values indexed either by mtlearn's
// morphological-tree node ids or by the Higra-compatible hierarchy exported by
// the backend.
enum class NodeIdSpace {
    MORPHOLOGICAL_TREE,
    HIGRA,
};

// Public attribute identifiers supported by the current morphology backend.
// Keep this enum synchronized with the conversion table in
// bindings/morphology/BindingSupport.hpp and the Python exposure in
// bindings/morphology/AttributeBinding.hpp.
enum class Attribute {
    AREA,
    VOLUME,
    RELATIVE_VOLUME,
    LEVEL,
    GRAY_HEIGHT,
    MEAN_LEVEL,
    VARIANCE_LEVEL,
    BOX_WIDTH,
    BOX_HEIGHT,
    DIAGONAL_LENGTH,
    RECTANGULARITY,
    RATIO_WH,
    BOX_COL_MIN,
    BOX_COL_MAX,
    BOX_ROW_MIN,
    BOX_ROW_MAX,
    CENTRAL_MOMENT_20,
    CENTRAL_MOMENT_02,
    CENTRAL_MOMENT_11,
    CENTRAL_MOMENT_30,
    CENTRAL_MOMENT_03,
    CENTRAL_MOMENT_21,
    CENTRAL_MOMENT_12,
    HU_MOMENT_1,
    HU_MOMENT_2,
    HU_MOMENT_3,
    HU_MOMENT_4,
    HU_MOMENT_5,
    HU_MOMENT_6,
    HU_MOMENT_7,
    INERTIA,
    COMPACTNESS,
    ECCENTRICITY,
    LENGTH_MAJOR_AXIS,
    LENGTH_MINOR_AXIS,
    AXIS_ORIENTATION,
    CIRCULARITY,
    BITQUADS_AREA,
    BITQUADS_NUMBER_EULER,
    BITQUADS_NUMBER_HOLES,
    BITQUADS_PERIMETER,
    BITQUADS_PERIMETER_CONTINUOUS,
    BITQUADS_CIRCULARITY,
    BITQUADS_PERIMETER_AVERAGE,
    BITQUADS_LENGTH_AVERAGE,
    BITQUADS_WIDTH_AVERAGE,
    HEIGHT_NODE,
    DEPTH_NODE,
    IS_LEAF_NODE,
    IS_ROOT_NODE,
    NUM_CHILDREN_NODE,
    NUM_SIBLINGS_NODE,
    NUM_DESCENDANTS_NODE,
    NUM_LEAF_DESCENDANTS_NODE,
    LEAF_RATIO_NODE,
    BALANCE_NODE,
    MAX_DIST,
    AVG_CHILD_HEIGHT_NODE,
};

// Attribute groups are convenience requests expanded by the backend attribute
// computer. They are part of the facade because Python notebooks and future
// C++ consumers should not depend on mmcfilters::AttributeGroup directly.
enum class AttributeGroup {
    ALL,
    GEOMETRIC,
    MOMENT_BASED,
    BOUNDING_BOX,
    CENTRAL_MOMENTS,
    HU_MOMENTS,
    TEXTURE,
    TREE_TOPOLOGY,
    BITQUADS,
};

using AttributeOrGroup = std::variant<Attribute, AttributeGroup>;

// Non-owning image view used at the C++ API boundary. Callers must keep
// `data` alive for the duration of the tree-construction call. The tree owns
// its own backend representation after construction.
struct ImageViewUInt8 {
    const std::uint8_t* data{nullptr};
    int rows{0};
    int cols{0};
};

// Owning row-major uint8 image returned by reconstruction routines. This type
// keeps the public API independent from the backend image container.
struct UInt8Image {
    int rows{0};
    int cols{0};
    std::vector<std::uint8_t> pixels;
};

// Copyable handle around the backend weighted morphological tree.
//
// WeightedTree is deliberately a small shared handle instead of exposing the
// backend object by value. This keeps Python bindings, CFP helpers, and future
// C++ consumers aligned around one stable facade while preserving cheap copies.
class WeightedTree {
public:
    WeightedTree(const WeightedTree&) noexcept = default;
    WeightedTree& operator=(const WeightedTree&) noexcept = default;
    WeightedTree(WeightedTree&&) noexcept = default;
    WeightedTree& operator=(WeightedTree&&) noexcept = default;
    ~WeightedTree();

    // Build a max-tree or min-tree from a 2D uint8 image. `radius` is forwarded
    // to the backend adjacency relation used by component-tree construction.
    static WeightedTree createComponentTree(ImageViewUInt8 image, bool isMaxTree, double radius = 1.5);

    // Build a tree of shapes from a 2D uint8 image. The infinity seed controls
    // the backend's boundary handling and should normally remain at the default
    // unless the caller needs exact compatibility with a previous experiment.
    static WeightedTree createTreeOfShapes(
        ImageViewUInt8 image,
        TreeOfShapesInterpolation interpolation = TreeOfShapesInterpolation::SelfDual,
        int infinitySeedRow = TreeOfShapesDefaultInfinityRow,
        int infinitySeedCol = TreeOfShapesDefaultInfinityCol);

    int numRows() const;
    int numCols() const;
    int numNodes() const;
    int numInternalNodeSlots() const;

    // Node-level values are indexed by backend/morphological-tree node id.
    float getAltitude(NodeId nodeId) const;
    float getNodeResidue(NodeId nodeId) const;

    // Mutating operations preserve the same backend semantics as mmcfilters.
    // They are intentionally exposed through the facade so Python notebooks do
    // not need direct access to the backend tree.
    void pruneNode(NodeId nodeId);
    void mergeNodeIntoParent(NodeId nodeId);

    // Reconstruct the current image represented by the tree after any pruning
    // or merging operations.
    UInt8Image reconstructionImage() const;

    // Export parent/altitude vectors compatible with Higra-style hierarchies.
    std::pair<std::vector<NodeId>, std::vector<float>> exportHigraHierarchy() const;

private:
    struct Impl;

    explicit WeightedTree(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> impl_;

    friend struct detail::BackendAccess;
};

using WeightedTreePtr = std::shared_ptr<WeightedTree>;

// Short free functions used by internal CFP code to keep formulas readable.
inline float altitude(const WeightedTree& tree, NodeId nodeId)
{
    return tree.getAltitude(nodeId);
}

inline float residue(const WeightedTree& tree, NodeId nodeId)
{
    return tree.getNodeResidue(nodeId);
}

} // namespace mtlearn::morphology
