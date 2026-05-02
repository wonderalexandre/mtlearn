# ---------------------------------------------------------------------------
# pybind11 module configuration (_mtlearn)
# ---------------------------------------------------------------------------

find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.1.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)

if(MTLEARN_WITH_TORCH)
    find_package(Torch REQUIRED)
    pybind11_add_module(_mtlearn MtlearnModule.cpp)
    # Filter Torch libraries that CMake discovered as NOTFOUND before linking.
    if(UNIX AND NOT APPLE)
        set(FILTERED_TORCH_LIBRARIES "")
        foreach(lib ${TORCH_LIBRARIES})
            if(NOT lib MATCHES "-NOTFOUND$")
                list(APPEND FILTERED_TORCH_LIBRARIES ${lib})
            endif()
        endforeach()
        target_link_libraries(_mtlearn PRIVATE mtlearn::core mmcfilters::core ${FILTERED_TORCH_LIBRARIES})
    elseif(APPLE)
        target_link_libraries(_mtlearn PRIVATE mtlearn::core mmcfilters::core ${TORCH_LIBRARIES})
    endif()

    get_filename_component(_torch_root "${TORCH_LIBRARY}" DIRECTORY)
    find_library(TORCH_PYTHON_LIBRARY torch_python
        HINTS "${_torch_root}" "${_torch_root}/.." "${_torch_root}/../.."
        PATH_SUFFIXES lib)
    if(TORCH_PYTHON_LIBRARY)
        target_link_libraries(_mtlearn PRIVATE "${TORCH_PYTHON_LIBRARY}")
    endif()
    target_include_directories(_mtlearn PRIVATE
        ${PROJECT_SOURCE_DIR}/mtlearn/include
        ${PROJECT_SOURCE_DIR}/mtlearn/src
        ${CMAKE_SOURCE_DIR}/external/MorphologicalAttributeFilters
        ${TORCH_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/external/MorphologicalAttributeFilters/pybinds)
    target_compile_definitions(_mtlearn PRIVATE MTLEARN_WITH_TORCH=1)
endif()

set_target_properties(_mtlearn PROPERTIES OUTPUT_NAME "_mtlearn")
set_property(TARGET _mtlearn PROPERTY CXX_STANDARD 20)
set_property(TARGET _mtlearn PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET _mtlearn PROPERTY CXX_EXTENSIONS OFF)
set_target_properties(_mtlearn PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS _mtlearn
        LIBRARY DESTINATION mtlearn
        ARCHIVE DESTINATION mtlearn
        RUNTIME DESTINATION mtlearn)
