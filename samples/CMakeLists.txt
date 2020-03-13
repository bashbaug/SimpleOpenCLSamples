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

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(SAMPLES_PLATFORM_NAME "x64")
elseif (CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(SAMPLES_PLATFORM_NAME "x86")
else()
    set(SAMPLES_PLATFORM_NAME "Unknown")
endif()

set(SAMPLES_CL_VERSION 120)

add_subdirectory( 00_enumopencl )
add_subdirectory( 00_enumopenclpp )
add_subdirectory( 01_copybuffer )
add_subdirectory( 02_copybufferkernel )
add_subdirectory( 03_mandelbrot )
add_subdirectory( 04_julia )
add_subdirectory( 05_kernelfromfile )

set(SAMPLES_CL_VERSION 210)
add_subdirectory( 05_spirvkernelfromfile )

add_subdirectory( usm )
