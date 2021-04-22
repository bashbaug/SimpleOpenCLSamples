#!/usr/bin/env python

import pyopencl as cl

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
