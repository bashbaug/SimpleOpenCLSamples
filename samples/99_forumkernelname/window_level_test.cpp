/**************************************************************************************************
    Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved.
**************************************************************************************************/
//#include <boost/test/unit_test.hpp>

#include "CL/cl.h"
#include <iostream>
#include <fstream>
#include <vector>

#define MAX_PLATFORM_SIZE 256
#define MAX_DEVICE_SIZE 256

namespace {
std::vector<unsigned char> load_file(const std::string& file)
{
    std::fstream input(file, std::ios::in | std::ios::binary | std::ios::ate);
    auto size = input.tellg();
    input.seekg(0, std::ios::beg);
    std::vector<unsigned char> binary(size);
    input.read((char*)binary.data(),size);
    input.close();

    return binary;
}
}

int main()
{
    std::vector<unsigned char> data = load_file("window_level.spv");

    cl_device_id device_id[256];
    cl_platform_id platform_id[256];
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    char buf[4096];

    cl_int ret = clGetPlatformIDs(0, 0, &ret_num_platforms);
    ret = clGetPlatformIDs(ret_num_platforms, platform_id, &ret_num_platforms);
    if (ret != CL_SUCCESS) {
        std::cout << "something went wrong in clGetPlatformIDs" << std::endl;
        return 0;
    }

    for (unsigned int i=0; i<ret_num_platforms; i++)
    {
        ret = clGetPlatformInfo(platform_id[i], CL_PLATFORM_NAME, sizeof(buf), buf, NULL);
        if (ret != CL_SUCCESS) {
            std::cout << "something went wrong in clGetPlatformInfo" << std::endl;
            return 1;
        }

        ret = clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_ALL, MAX_DEVICE_SIZE, device_id, &ret_num_devices);
        if (ret != CL_SUCCESS) {
            std::cout << "something went wrong in clGetDeviceIDs" << std::endl;
            return 1;
        }

        for (unsigned int j=0; j<ret_num_devices; j++) {
            ret = clGetDeviceInfo(device_id[j], CL_DEVICE_NAME, sizeof(buf), buf, NULL);
            if (ret != CL_SUCCESS) {
                std::cout << "something went wrong in clGetDeviceInfo" << std::endl;
                return 1;
            }

            ret = clGetDeviceInfo(device_id[j], CL_DEVICE_VENDOR, sizeof(buf), buf, NULL);
            if (ret != CL_SUCCESS) {
                std::cout << "something went wrong in clGetDeviceInfo" << std::endl;
                return 1;
            }
        }
        std::cout << std::endl;
    }

    cl_context context = clCreateContext( NULL, 1, &device_id[0], NULL, NULL, &ret);

    cl_int err = 0;
    cl_program program = clCreateProgramWithIL((cl_context)context, data.data(), data.size(), &err);
    if(err != CL_SUCCESS){
        std::cout << "failed to create opencl program with IL " << std::endl;
        return 1;
    }

    err = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);

    cl_int build_status;
    err = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);

    if(build_status != CL_SUCCESS){
        size_t ret_val_size;
        err = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);

        std::vector<char> build_log(ret_val_size+1,0);
        err = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log.data(), NULL);

        build_log[ret_val_size] = '\0';
        std::cout << "failed to build opencl program: \n" << build_log.data() << std::endl;
    }


    size_t ret_val_size = 0;
    err = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, 0, NULL, &ret_val_size);

    std::vector<char> kernel_names(ret_val_size+1,0);
    err = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, ret_val_size, kernel_names.data(), NULL);

    kernel_names[ret_val_size] = '\0';
    std::cout << "kernel names: \n" << kernel_names.data() << std::endl;

    // prints out window_level.1

    return 0;
}
