#!/usr/bin/env python

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

    devices = platforms[platformIndex].get_device()
    print('Running on device: ' + devices[deviceIndex].get_info(cl.device_info.NAME))

    context = cl.Context(devices[deviceIndex])
    commandQueue = cl.CommandQueue(context, devices[deviceIndex])

    deviceMemSrc = cl.Buffer(context, cl.mem_flags.ALLOC_HOST_PTR, )

for platform in cl.get_platforms():
    print("Platform:")
    print("        Name:           " + platform.get_info(cl.platform_info.NAME))
    print("        Vendor:         " + platform.get_info(cl.platform_info.VENDOR))
    print("        Driver Version: " + platform.get_info(cl.platform_info.VERSION))
    for i, device in enumerate(platform.get_devices()):
        print("Device[{}]:".format(i))
        print("        Type:           " + cl.device_type.to_string(device.get_info(cl.device_info.TYPE)))
        print("        Name:           " + device.get_info(cl.device_info.NAME))
        print("        Vendor:         " + device.get_info(cl.device_info.VENDOR))
        print("        Device Version: " + device.get_info(cl.device_info.VERSION))
        print("        Driver Version: " + device.get_info(cl.device_info.DRIVER_VERSION))
    print()
