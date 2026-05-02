#pragma once

// Pybind wrapper for CFP tensor extraction.
//
// The wrapped implementation lives in mtlearn/src/mtlearn/cfp/tree_tensors.hpp.
// This file should only describe Python names, argument names, docstrings, and
// pybind call policy.

#include "mtlearn/cfp/tree_tensors.hpp"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace mtlearn {

namespace py = pybind11;

// Expose stateless tensor helpers as a Python class to preserve the historical
// mtlearn API shape used by notebooks and layer implementations.
inline void init_ConnectedFilterPreprocessingTreeTensors(py::module& m)
{
    using TreeTensors = cfp::ConnectedFilterPreprocessingTreeTensors;

    py::class_<TreeTensors>(m, "ConnectedFilterPreprocessingTreeTensors")
        .def_static(
            "get_residues",
            &TreeTensors::getResidues,
            py::arg("tree"),
            "Return the residues of all tree nodes as a tensor.")
        .def_static(
            "get_jacobian",
            &TreeTensors::getJacobian,
            py::arg("tree"),
            "Return the sparse tree Jacobian as a tensor.")
        .def_static(
            "get_info_for_jacobian",
            &TreeTensors::getInfoForJacobian,
            py::arg("tree"),
            py::call_guard<py::gil_scoped_release>(),
            "Return residues and implicit-Jacobian helper tensors.");
}

} // namespace mtlearn
