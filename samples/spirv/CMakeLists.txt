# Copyright (c) 2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

include(FindPkgConfig)

# first try locating SPIRV-Tools via pkgconfig (the old way)
pkg_search_module(SPIRV_TOOLS SPIRV-Tools)
if (NOT SPIRV_TOOLS_FOUND)
  # then try locating SPIRV-Tools via cmake (the new way)
  find_package(SPIRV-Tools)
  find_package(SPIRV-Tools-tools)
  if (SPIRV-Tools_FOUND AND SPIRV-Tools-tools_FOUND)
    set(SPIRV_TOOLS_FOUND TRUE)
    # check for the existence of library targets in the found packages
    if(TARGET SPIRV-Tools-shared)
      # use the shared libary target if present
      set(SPIRV-Tools-library SPIRV-Tools-shared)
    elseif(TARGET SPIRV-Tools-static)
      # otherwise fallback to the static library target
      set(SPIRV-Tools-library SPIRV-Tools-static)
    else()
      message(FATAL_ERROR "Found SPIRV-Tools package but neither "
              "SPIRV-Tools-shared or SPIRV-Tools-static targets exist.")
    endif()
    set(SPIRV_TOOLS_LDFLAGS ${SPIRV-Tools-library})
    get_target_property(SPIRV_TOOLS_INCLUDE_DIRS ${SPIRV-Tools-library} INTERFACE_INCLUDE_DIRECTORIES)
  endif()
endif()

if (SPIRV_TOOLS_FOUND)
    add_subdirectory( 01_spirvprobe )
else()
    message(STATUS "SPIR-V tools are not found; some samples will not be built.")
endif()
