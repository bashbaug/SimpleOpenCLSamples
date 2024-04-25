#!/usr/bin/env python

# Copyright (c) 2019-2023 Ben Ashbaugh
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
import time

filename = 'julia.bmp'

iterations = 16
gwx = 512
gwy = 512
lwx = 0
lwy = 0

cr = -0.123
ci = 0.745

kernelString = """
kernel void Julia( global uchar4* dst, float cr, float ci )
{
    const float cMinX = -1.5f;
    const float cMaxX =  1.5f;
    const float cMinY = -1.5f;
    const float cMaxY =  1.5f;

    const int cWidth = get_global_size(0);
    const int cIterations = 16;

    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);

    float a = x * ( cMaxX - cMinX ) / cWidth + cMinX;
    float b = y * ( cMaxY - cMinY ) / cWidth + cMinY;

    float result = 0.0f;
    const float thresholdSquared = cIterations * cIterations / 64.0f;

    for( int i = 0; i < cIterations; i++ ) {
        float aa = a * a;
        float bb = b * b;

        float magnitudeSquared = aa + bb;
        if( magnitudeSquared >= thresholdSquared ) {
            break;
        }

        result += 1.0f / cIterations;
        b = 2 * a * b + ci;
        a = aa - bb + cr;
    }

    result = max( result, 0.0f );
    result = min( result, 1.0f );

    // RGBA
    float4 color = (float4)( result, sqrt(result), 1.0f, 1.0f );

    dst[ y * cWidth + x ] = convert_uchar4(color * 255.0f);
}
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--platform', type=int, action='store', default=0, help='Platform Index')
    parser.add_argument('-d', '--device', type=int, action='store', default=0, help='Device Index')
    parser.add_argument('-i', '--iterations', type=int, action='store', default=iterations, help='Iterations')
    parser.add_argument('--gwx', type=int, action='store', default=gwx, help='Global Work Size X AKA Image Width')
    parser.add_argument('--gwy', type=int, action='store', default=gwy, help='Global Work Size X AKA Image Width')
    parser.add_argument('--lwx', type=int, action='store', default=lwx, help='Local Work Size X')
    parser.add_argument('--lwy', type=int, action='store', default=lwy, help='Local Work Size Y')

    args = parser.parse_args()
    platformIndex = args.platform
    deviceIndex = args.device
    iterations = args.iterations
    gwx = args.gwx
    gwy = args.gwy
    lwx = args.lwx
    lwy = args.lwy

    platforms = cl.get_platforms()
    print('Running on platform: ' + platforms[platformIndex].get_info(cl.platform_info.NAME))

    devices = platforms[platformIndex].get_devices()
    print('Running on device: ' + devices[deviceIndex].get_info(cl.device_info.NAME))

    context = cl.Context([devices[deviceIndex]])
    commandQueue = cl.CommandQueue(context, devices[deviceIndex])

    program = cl.Program(context, kernelString)
    program.build()
    kernel = program.Julia

    deviceMemDst = cl.Buffer(context, cl.mem_flags.ALLOC_HOST_PTR, 
                             gwx * gwy * 4 * np.uint8().itemsize)

    lws = None

    print('Executing the kernel {} times'.format(iterations))
    print('Global Work Size = ({}, {})'.format(gwx, gwy))
    if lwx > 0 and lwy > 0:
        print('Local Work Size = ({}, {})'.format(lwx, lwy))
        lws = [lwx, lwy]
    else:
        print('Local Work Size = NULL')

    # Ensure the queue is empty and no processing is happening
    # on the device before starting the timer.
    commandQueue.finish()

    start = time.perf_counter()
    for i in range(iterations):
        kernel(commandQueue, [gwx, gwy], lws,
               deviceMemDst, np.float32(cr), np.float32(ci))

    # Ensure all processing is complete before stopping the timer.
    commandQueue.finish()

    end = time.perf_counter()
    print('Finished in {} seconds'.format(end - start))

    mapped_dst, event = cl.enqueue_map_buffer(commandQueue, deviceMemDst,
                                              cl.map_flags.READ, 
                                              0, gwx * gwy, np.uint32)
    with mapped_dst.base:
        # note: this generates a 24-bit .bmp file instead of a 32-bit .bmp file!
        (r, g, b, a) = Image.fromarray(mapped_dst.reshape((gwy, gwx)), 'RGBA').split()
        image = Image.merge('RGB', (r, g, b))
        image.save(filename)
        print('Wrote image file {}'.format(filename))
