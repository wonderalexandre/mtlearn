import mtlearn
import pytest

if not getattr(mtlearn, "WITH_TORCH", False):
    pytest.skip("build has no LibTorch support", allow_module_level=True)

try:
    import numpy as np
    import torch
    from torch.autograd import gradcheck
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Python dependency unavailable: {exc}", allow_module_level=True)

from mtlearn import morphology

pytestmark = [pytest.mark.gradcheck, pytest.mark.slow]


def _small_gradcheck_case(dtype, tree_type="max-tree", tos_interpolation=None):
    image = np.array(
        [
            [2, 2, 0],
            [2, 5, 0],
            [3, 3, 1],
        ],
        dtype=np.uint8,
    )
    tree = morphology.build_tree(
        image,
        tree_type,
        tos_interpolation=tos_interpolation,
    )
    attributes = morphology.compute_attributes(
        tree,
        [morphology.AttributeType.AREA, morphology.AttributeType.COMPACTNESS],
    )[1]
    attributes = (attributes.min() - attributes) / (
        attributes - attributes.max() + 1e-8
    )
    return tree, torch.as_tensor(attributes, dtype=dtype)


def _learnable_parameters(dtype):
    weight = torch.tensor([0.35, -0.2], dtype=dtype, requires_grad=True)
    bias = torch.tensor([0.1], dtype=dtype, requires_grad=True)
    return weight, bias


def test_explicit_jacobian_function_gradcheck():
    tree, attributes = _small_gradcheck_case(torch.float64)
    jacobian = mtlearn.ConnectedFilterPreprocessingTreeTensors.get_jacobian(tree).to(
        dtype=torch.float64
    )
    residues = mtlearn.ConnectedFilterPreprocessingTreeTensors.get_residues(tree).to(
        dtype=torch.float64
    )
    weight, bias = _learnable_parameters(torch.float64)

    def filtered_mean(w, b):
        return mtlearn.layers.ConnectedFilterPreprocessingExplicitJacobianFunction.apply(
            jacobian,
            residues,
            tree.numRows,
            tree.numCols,
            attributes,
            w,
            b,
            1.0,
            False,
        ).mean()

    assert gradcheck(filtered_mean, (weight, bias), eps=1e-6, atol=1e-4)


def test_implicit_jacobian_function_gradcheck():
    tree, attributes = _small_gradcheck_case(torch.float64)
    residues, tpre, tpost, parent, node_of_pixel = (
        mtlearn.ConnectedFilterPreprocessingTreeTensors.get_info_for_jacobian(tree)
    )
    residues = residues.to(dtype=torch.float64)
    weight, bias = _learnable_parameters(torch.float64)

    def filtered_mean(w, b):
        return mtlearn.layers.ConnectedFilterPreprocessingImplicitJacobianFunction.apply(
            w,
            b,
            residues,
            tpre,
            tpost,
            parent,
            node_of_pixel,
            attributes,
            tree.numRows,
            tree.numCols,
            1.0,
            False,
        ).mean()

    assert gradcheck(filtered_mean, (weight, bias), eps=1e-6, atol=1e-4)


def test_implicit_jacobian_function_gradcheck_tree_of_shapes():
    tree, attributes = _small_gradcheck_case(
        torch.float64,
        tree_type="tree-of-shapes",
        tos_interpolation="self-dual",
    )
    residues, tpre, tpost, parent, node_of_pixel = (
        mtlearn.ConnectedFilterPreprocessingTreeTensors.get_info_for_jacobian(tree)
    )
    residues = residues.to(dtype=torch.float64)
    weight, bias = _learnable_parameters(torch.float64)

    def filtered_mean(w, b):
        return mtlearn.layers.ConnectedFilterPreprocessingImplicitJacobianFunction.apply(
            w,
            b,
            residues,
            tpre,
            tpost,
            parent,
            node_of_pixel,
            attributes,
            tree.numRows,
            tree.numCols,
            1.0,
            False,
        ).mean()

    assert gradcheck(filtered_mean, (weight, bias), eps=1e-6, atol=1e-4)


def test_cpu_tree_traversal_function_matches_numeric_gradient():
    tree, attributes = _small_gradcheck_case(torch.float32)
    weight, bias = _learnable_parameters(torch.float32)
    eps = 3e-4
    atol = 1e-3

    def filtered_mean(w, b):
        return mtlearn.layers.ConnectedFilterPreprocessingCPUTreeTraversalFunction.apply(
            tree,
            attributes,
            w,
            b,
            1.0,
            False,
        ).mean()

    output = filtered_mean(weight, bias)
    output.backward()

    numeric_weight = []
    for index in range(weight.numel()):
        weight_increased = weight.detach().clone()
        weight_decreased = weight.detach().clone()
        weight_increased[index] += eps
        weight_decreased[index] -= eps
        numeric_weight.append(
            (
                filtered_mean(weight_increased, bias.detach())
                - filtered_mean(weight_decreased, bias.detach())
            )
            / (2 * eps)
        )

    bias_increased = bias.detach().clone()
    bias_decreased = bias.detach().clone()
    bias_increased[0] += eps
    bias_decreased[0] -= eps
    numeric_bias = (
        filtered_mean(weight.detach(), bias_increased)
        - filtered_mean(weight.detach(), bias_decreased)
    ) / (2 * eps)

    assert torch.allclose(weight.grad[0], numeric_weight[0], atol=atol)
    assert torch.allclose(weight.grad[1], numeric_weight[1], atol=atol)
    assert torch.allclose(bias.grad[0], numeric_bias, atol=atol)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
