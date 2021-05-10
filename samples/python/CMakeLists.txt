# Copyright (c) 2019-2021 Ben Ashbaugh
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

function(add_opencl_python_sample)
    set(options )
    set(one_value_args NUMBER CATEGORY)
    set(multi_value_args SOURCES KERNELS)
    cmake_parse_arguments(OPENCL_PYTHON_SAMPLE
        "${options}" "${one_value_args}" "${multi_value_args}"
        ${ARGN}
    )

    if(NOT OPENCL_PYTHON_SAMPLE_NUMBER)
        message(STATUS "No sample number specified for python sample ${OPENCL_PYTHON_SAMPLE_TARGET}, using 99.")
        set(OPENCL_PYTHON_SAMPLE_NUMBER 99)
    endif()

    set_target_properties(${OPENCL_PYTHON_SAMPLE_TARGET} PROPERTIES FOLDER "Samples/Python/${OPENCL_PYTHO_SAMPLE_CATEGORY}/${OPENCL_PYTHON_SAMPLE_NUMBER}_${OPENCL_PYTHON_SAMPLE_TARGET}")

    if(CMAKE_CONFIGURATION_TYPES)
        set(OPENCL_PYTHON_SAMPLE_CONFIGS ${CMAKE_CONFIGURATION_TYPES})
    else()
        set(OPENCL_PYTHON_SAMPLE_CONFIGS ${CMAKE_BUILD_TYPE})
    endif()
    foreach(CONFIG ${OPENCL_PYTHON_SAMPLE_CONFIGS})
        install(FILES ${OPENCL_PYTHON_SAMPLE_SOURCES} CONFIGURATIONS ${CONFIG} DESTINATION ${CONFIG})
        install(FILES ${OPENCL_PYTHON_SAMPLE_KERNELS} CONFIGURATIONS ${CONFIG} DESTINATION ${CONFIG})
    endforeach()
endfunction()

add_subdirectory( 00_enumopencl )
add_subdirectory( 01_copybuffer )
add_subdirectory( 02_copybufferkernel )
add_subdirectory( 03_mandelbrot )
add_subdirectory( 04_julia )
