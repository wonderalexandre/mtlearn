// Optional embedded-Python import test for the native mtlearn package.
//
// CTest normally keeps this disabled through MTLEARN_ENABLE_EMBED=0 because
// embedding Python can be environment-sensitive on developer machines and CI.
// When explicitly enabled, the test verifies that the built Python package and
// native extension can be imported from a C++ embedded interpreter.

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

// MTLEARN_EMBED_DEFAULT is defined by CMake (0 = disabled, 1 = run the test
// automatically). Users can override it through MTLEARN_ENABLE_EMBED.

namespace py = pybind11;

#ifndef MTLEARN_EMBED_DEFAULT
#define MTLEARN_EMBED_DEFAULT 0
#endif

int main()
{
#if !MTLEARN_WITH_TORCH
    std::cout << "SKIP embed test: build has no LibTorch support" << std::endl;
    return 0;
#else
    try {
        constexpr bool kEmbedDefault = MTLEARN_EMBED_DEFAULT != 0;
        bool should_embed = kEmbedDefault;
        if (const char* flag = std::getenv("MTLEARN_ENABLE_EMBED")) {
            should_embed = std::string(flag) != "0";
        }

        if (!should_embed) {
            std::cout << "SKIP embed test disabled" << std::endl;
            return 0;
        }

        py::scoped_interpreter guard{};

        // Prefer the just-built Python package and extension over any globally
        // installed mtlearn so this test validates the current build tree.
        auto sys = py::module::import("sys");
        auto path = sys.attr("path");

        std::filesystem::path python_dir = std::filesystem::weakly_canonical(MTLEARN_PYTHON_DIR);
        std::filesystem::path bindings_dir = std::filesystem::weakly_canonical(MTLEARN_BINDINGS_DIR);

        path.attr("insert")(0, python_dir.string());
        path.attr("insert")(0, bindings_dir.string());

        try {
            // Torch import is required before importing the extension on some
            // platforms because the native module links against LibTorch.
            py::module::import("torch");
        } catch (const py::error_already_set& e) {
            if (e.matches(PyExc_ImportError)) {
                std::cout << "SKIP torch unavailable: " << e.what() << std::endl;
                return 0;
            }
            throw;
        }

        py::module::import("mtlearn");
        py::module::import("mtlearn.morphology");
        py::module::import("mtlearn.layers");

        auto mtlearn = py::module::import("mtlearn");
        if (!mtlearn.attr("WITH_TORCH").cast<bool>()) {
            std::cerr << "Expected mtlearn.WITH_TORCH to be true" << std::endl;
            return 1;
        }

        std::cout << "mtlearn embedded import succeeded" << std::endl;
        return 0;
    }
    catch (const py::error_already_set& e) {
        std::cerr << "Python exception:\n" << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "C++ exception:\n" << e.what() << std::endl;
        return 1;
    }
#endif
}
