# Copyright (c) 2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

find_package(Python3 COMPONENTS Interpreter QUIET)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/spirv_capability_deps.def
    COMMENT "Generating spirv_capability_deps.def..."
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_spirv_capability_deps.py
            --grammar "${SPIRV-Headers_SOURCE_DIR}/include/spirv/unified1/spirv.core.grammar.json"
            --output "${CMAKE_CURRENT_BINARY_DIR}/spirv_capability_deps.def"
    DEPENDS generate_spirv_capability_deps.py "${SPIRV-Headers_SOURCE_DIR}/include/spirv/unified1/spirv.core.grammar.json"
    USES_TERMINAL
    VERBATIM)

add_opencl_sample(
    TEST
    NUMBER 01
    TARGET spirvquerycheck
    VERSION 300
    CATEGORY spirv
    SOURCES main.cpp ${CMAKE_CURRENT_BINARY_DIR}/spirv_capability_deps.def
    INCLUDES ${SPIRV-Headers_SOURCE_DIR}/include ${SPIRV_TOOLS_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR}
    LIBS ${SPIRV_TOOLS_LDFLAGS})
