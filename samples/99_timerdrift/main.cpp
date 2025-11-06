/*
// Copyright (c) 2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <iostream>
#include <chrono>
#include <ctime>
#include <thread>

#include <cinttypes>

void getGlobalTimeStamps(cl::Device& device,
    uint64_t *DeviceTimestamp,
    uint64_t *HostTimestamp)
{
    auto times = device.getDeviceAndHostTimer();

    if (DeviceTimestamp) {
        *DeviceTimestamp = times.first;
    }
    if (HostTimestamp) {
        *HostTimestamp = times.second;
    }
}

int main(int argc, char** argv)
{
    int platformIndex = 0;
    int deviceIndex = 0;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: timerdrift [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    uint64_t BaseHostTime = 0, CurrentHostTime = 0, BaseDeviceTime = 0, CurrentDeviceTime = 0;

    for (int i = 0; i < 32; i++) {
        if (i == 0) {
            getGlobalTimeStamps(devices[deviceIndex], &BaseDeviceTime, &BaseHostTime);
            CurrentDeviceTime = BaseDeviceTime;
            CurrentHostTime = BaseHostTime;
        } else {
            getGlobalTimeStamps(devices[deviceIndex], &CurrentDeviceTime, &CurrentHostTime);
        }
        auto DeviceTimePast = CurrentDeviceTime - BaseDeviceTime;
        auto HostTimePast = CurrentHostTime - BaseHostTime;
#if 0
        std::cout << "Iteration: " << i << std::endl;
        std::cout << "Device time past since base time: " << DeviceTimePast
                    << std::endl;
        std::cout << "Host time past since base time: " << HostTimePast
                    << std::endl;
        std::cout << "Abs diff: "
                    << std::abs((int64_t)HostTimePast - (int64_t)DeviceTimePast)
                    << std::endl
                    << std::endl;
#else
        printf("Iteration: %2d, Delta: %" PRId64 "\n",
            i, (int64_t)HostTimePast - (int64_t)DeviceTimePast);
#endif

        // Emulate the first iteration taking longer
        if (i == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    return 0;
}