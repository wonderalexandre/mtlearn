# ---------------------------------------------------------------------------
# Main C++ target for the public mtlearn morphology facade.
# ---------------------------------------------------------------------------

# C++ core implementation files.
set(MTLEARN_CORE_SOURCES
    mtlearn/morphology.cpp
)

add_library(mtlearn_core ${MTLEARN_CORE_SOURCES})
add_library(mtlearn::core ALIAS mtlearn_core)
set_target_properties(mtlearn_core PROPERTIES EXPORT_NAME core)

# Export the public include tree for builds and installed consumers. Backend
# detail headers remain excluded from installation below.
target_include_directories(mtlearn_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/mtlearn/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${MTLEARN_MMCFILTERS_DIR}
        ${MTLEARN_MMCFILTERS_DIR}/mmcfilters
)

target_compile_features(mtlearn_core PUBLIC cxx_std_20)

# The public C++ morphology facade uses the backend only in implementation
# files.
if(MTLEARN_ENABLE_ASSERTS)
    target_compile_definitions(mtlearn_core PRIVATE MMCFILTERS_ENABLE_ASSERTS)
    target_compile_options(mtlearn_core
        PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
else()
    target_compile_definitions(mtlearn_core PRIVATE NDEBUG)
endif()

# Install rules for consumers using mtlearn as an external C++ library.
install(TARGETS mtlearn_core
        EXPORT mtlearnTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/mtlearn/include/mtlearn
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
            PATTERN "*.hpp"
            PATTERN "detail" EXCLUDE
            PATTERN "detail/*" EXCLUDE)
