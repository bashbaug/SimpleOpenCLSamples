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

import numpy as np
import pyopencl as cl
import argparse

gwx = 1024 * 1024

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

    deviceMemSrc = cl.Buffer(context, cl.mem_flags.ALLOC_HOST_PTR, gwx * np.uint32().itemsize)
    deviceMemDst = cl.Buffer(context, cl.mem_flags.ALLOC_HOST_PTR, gwx * np.uint32().itemsize)

    # initialization
    mapped_src, event = cl.enqueue_map_buffer(commandQueue, deviceMemSrc,
                                              cl.map_flags.WRITE_INVALIDATE_REGION,
                                              0, gwx, np.uint32)
    with mapped_src.base:
        for i in range(gwx):
            mapped_src[i] = i

    # execution
    cl.enqueue_copy(commandQueue, deviceMemDst, deviceMemSrc)

    # verification
    mapped_dst, event = cl.enqueue_map_buffer(commandQueue, deviceMemDst,
                                              cl.map_flags.READ,
                                              0, gwx, np.uint32)
    with mapped_dst.base:
        mismatches = 0
        for i, val in enumerate(mapped_dst):
            if val != i:
                if mismatches < 16:
                    print('Mismatch!  dst[{}] == {}, want {}'.format(i, val, i))
                mismatches = mismatches + 1
        if mismatches != 0:
            print('Error: Found {} mismatches / {} values!!!'.format(mismatches, gwx))
        else:
            print('Success.')
