#!/usr/bin/env python

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

from PIL import Image

import numpy as np
import pyopencl as cl
import argparse
import PIL

filename = 'mandelbrot.bmp'

width = 768
height = 512

maxIterations = 256

kernelString = """
static inline int mandel(float c_re, float c_im, int count) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {
        if (z_re * z_re + z_im * z_im > 4.)
            break;

        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;

        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}
kernel void Mandelbrot(
    float x0, float y0,
    float x1, float y1,
    int width, int height,
    int maxIterations,
    global int* output)
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    float x = x0 + get_global_id(0) * dx;
    float y = y0 + get_global_id(1) * dy;

    int index = get_global_id(1) * width + get_global_id(0);
    output[index] = mandel(x, y, maxIterations);
}
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--platform', type=int, action='store', default=0, help='Platform Index')
    parser.add_argument('-d', '--device', type=int, action='store', default=0, help='Device Index')

    args = parser.parse_args()
    platformIndex = args.platform
    deviceIndex = args.device

    platforms = cl.get_platforms()
    print('Running on platform: ' + platforms[platformIndex].get_info(cl.platform_info.NAME))

    devices = platforms[platformIndex].get_devices()
    print('Running on device: ' + devices[deviceIndex].get_info(cl.device_info.NAME))

    context = cl.Context([devices[deviceIndex]])
    commandQueue = cl.CommandQueue(context, devices[deviceIndex])

    program = cl.Program(context, kernelString)
    program.build()
    kernel = program.Mandelbrot

    deviceMemDst = cl.Buffer(context, cl.mem_flags.ALLOC_HOST_PTR, 
                             width * height * np.uint32().itemsize)

    # execution
    kernel(commandQueue, [width, height], None, 
           np.float32(-2.0), np.float32(-1.0), np.float32(1.0), np.float32(1.0),
           np.int32(width), np.int32(height), np.int32(maxIterations), deviceMemDst)

    # save bitmap
    mapped_dst, event = cl.enqueue_map_buffer(commandQueue, deviceMemDst,
                                              cl.map_flags.READ, 
                                              0, width * height, np.uint32)
    with mapped_dst.base:
        colors = np.fromiter((240 if x & 1 else 20 for x in mapped_dst), np.uint8)
        image = Image.fromarray(colors.reshape((height, width)))
        image.save(filename)
        print('Wrote image file {}'.format(filename))
