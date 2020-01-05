# Copyright (c) 2020 Ben Ashbaugh
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

set(LIBUSM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include )
set(LIBUSM_INCLUDE_DIR ${LIBUSM_INCLUDE_DIR} PARENT_SCOPE)

set( LIBUSM_SOURCE_FILES
    include/libusm.h
    src/libusm.cpp
)

source_group( Source FILES ${LIBUSM_SOURCE_FILES} )

add_library( usm STATIC ${LIBUSM_SOURCE_FILES} )
target_include_directories( usm PRIVATE ${OpenCL_INCLUDE_DIR} ${LIBUSM_INCLUDE_DIR} )
target_compile_definitions( usm PRIVATE CL_TARGET_OPENCL_VERSION=120 )

set_target_properties( usm PROPERTIES FOLDER "Samples/USM/libusm" )
