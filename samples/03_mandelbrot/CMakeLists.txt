# Copyright (c) 2019-2020 Ben Ashbaugh
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

set(SAMPLE_NAME mandelbrot)
set(SAMPLE_NUMBER 03)

project(${SAMPLE_NAME})
add_executable( ${SAMPLE_NAME} main.cpp )

target_include_directories( ${SAMPLE_NAME} PRIVATE ${OpenCL_INCLUDE_DIR} )
target_link_libraries( ${SAMPLE_NAME} OpenCL )

target_compile_definitions( ${SAMPLE_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=${SAMPLES_CL_VERSION} )
target_compile_definitions( ${SAMPLE_NAME} PRIVATE CL_HPP_TARGET_OPENCL_VERSION=${SAMPLES_CL_VERSION})
target_compile_definitions( ${SAMPLE_NAME} PRIVATE CL_HPP_MINIMUM_OPENCL_VERSION=${SAMPLES_CL_VERSION})
if (WIN32)
    target_compile_definitions( ${SAMPLE_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

set_target_properties( ${SAMPLE_NAME} PROPERTIES FOLDER "Samples/${SAMPLE_NUMBER}_${SAMPLE_NAME}" )

if(CMAKE_CONFIGURATION_TYPES)
    set(SAMPLE_CONFIGS ${CMAKE_CONFIGURATION_TYPES})
else()
    set(SAMPLE_CONFIGS ${CMAKE_BUILD_TYPE})
endif()
foreach(CONFIG ${SAMPLE_CONFIGS})
    install(TARGETS ${SAMPLE_NAME} CONFIGURATIONS ${CONFIG} DESTINATION ${SAMPLES_PLATFORM_NAME}/${CONFIG})
endforeach()
