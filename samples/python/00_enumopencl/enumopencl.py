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

import pyopencl as cl

for p, platform in enumerate(cl.get_platforms()):
    print("Platform[{}]:".format(p))
    print("        Name:           " + platform.get_info(cl.platform_info.NAME))
    print("        Vendor:         " + platform.get_info(cl.platform_info.VENDOR))
    print("        Driver Version: " + platform.get_info(cl.platform_info.VERSION))
    for d, device in enumerate(platform.get_devices()):
        print("Device[{}]:".format(d))
        print("        Type:           " + cl.device_type.to_string(device.get_info(cl.device_info.TYPE)))
        print("        Name:           " + device.get_info(cl.device_info.NAME))
        print("        Vendor:         " + device.get_info(cl.device_info.VENDOR))
        print("        Device Version: " + device.get_info(cl.device_info.VERSION))
        print("        Driver Version: " + device.get_info(cl.device_info.DRIVER_VERSION))
    print()
