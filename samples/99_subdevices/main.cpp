/*
// Copyright (c) 2019-2020 Ben Ashbaugh
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
*/

#include <CL/opencl.hpp>

#define CASE_TO_STRING(_e) case _e: return #_e;

const char* partition_property_to_string(cl_device_partition_property property)
{
    switch (property) {
#ifdef CL_VERSION_1_2
    CASE_TO_STRING(CL_DEVICE_PARTITION_EQUALLY);
    CASE_TO_STRING(CL_DEVICE_PARTITION_BY_COUNTS);
    CASE_TO_STRING(CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN);
#endif
#ifdef cl_ext_device_fission
    CASE_TO_STRING(CL_DEVICE_PARTITION_EQUALLY_EXT);
    CASE_TO_STRING(CL_DEVICE_PARTITION_BY_COUNTS_EXT);
    CASE_TO_STRING(CL_DEVICE_PARTITION_BY_NAMES_EXT);
    CASE_TO_STRING(CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN_EXT);
#endif
    default: return "Unknown cl_device_partition_property";
    }
}

int main(
    int argc,
    char** argv )
{
    bool printUsage = false;
    int platformIndex = 0;
    int deviceIndex = 0;
    int partitionType = 0;

    if( argc < 1 )
    {
        printUsage = true;
    }
    else
    {
        for( size_t i = 1; i < argc; i++ )
        {
            if( !strcmp( argv[i], "-d" ) )
            {
                ++i;
                if( i < argc )
                {
                    deviceIndex = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-p" ) )
            {
                ++i;
                if( i < argc )
                {
                    platformIndex = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-pt" ) )
            {
                ++i;
                if( i < argc )
                {
                    partitionType = strtol(argv[i], NULL, 10);
                }
            }
            else
            {
                printUsage = true;
            }
        }
    }
    if( printUsage )
    {
        fprintf(stderr,
            "Usage: subdevices [options]\n"
            "Options:\n"
            "      -d: Device Index (default = 0)\n"
            "      -p: Platform Index (default = 0)\n"
            "      -pt: Partition Type (default = 0)\n"
            "               0 = by affinity domain\n"
            "               1 = equally\n"
            "               2 = by counts\n"
            );

        return -1;
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    std::vector<cl_device_partition_property> supported =
        devices[deviceIndex].getInfo<CL_DEVICE_PARTITION_PROPERTIES>();
    printf("Device supports %d partition properties.\n", (int)supported.size());
    for( auto& property : supported )
    {
        printf("%s (%04X)\n",
            partition_property_to_string(property),
            (unsigned)property);
    }

    cl_uint maxSubDevices =
        devices[deviceIndex].getInfo<CL_DEVICE_PARTITION_MAX_SUB_DEVICES>();
    printf("Device supports up to %d sub-devices.\n", maxSubDevices);

    cl_uint maxComputeUnits =
        devices[deviceIndex].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    printf("Device reports %d compute units.\n", maxComputeUnits);

    const cl_device_partition_property partitionByAffinity[] = {
        CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN, CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE, 0
    };
    const cl_device_partition_property partitionEqually[] = {
        CL_DEVICE_PARTITION_EQUALLY, 2, 0
    };
    const cl_device_partition_property partitionByCounts[] = {
        CL_DEVICE_PARTITION_BY_COUNTS, 1, 1, 0,
    };
    std::vector<cl::Device> subdevices;
    devices[deviceIndex].createSubDevices(
        partitionType == 0 ? partitionByAffinity : 
        partitionType == 1 ? partitionEqually :
        partitionType == 2 ? partitionByCounts :
        nullptr, &subdevices);

    printf("Partitioned into %d sub-devices.\n", (int)subdevices.size());

    if( subdevices.size() != 0)
    {
        printf("Creating a context with the first sub-device:\n");
        cl::Context context{subdevices[0]};
    }

    if( subdevices.size() != 0)
    {
        printf("Creating a context with all sub-devices:\n");
        cl::Context context{subdevices};
    }

    if( subdevices.size() != 0)
    {
        printf("Creating a context with the parent device and all sub-devices:\n");

        subdevices.push_back(devices[deviceIndex]);

        cl::Context context{subdevices};
    }

    return 0;
}