/*
// Copyright (c) 2019-2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>
#include "util.hpp"

#if !defined(CL_INTEL_SUBGROUP_MATRIX_MULTIPLY_ACCUMULATE_EXTENSION_NAME)
#define CL_INTEL_SUBGROUP_MATRIX_MULTIPLY_ACCUMULATE_EXTENSION_NAME \
    "cl_intel_subgroup_matrix_multiply_accumulate"
#endif

#if !defined(CL_INTEL_SUBGROUP_MATRIX_MULTIPLY_ACCUMULATE_TF32_EXTENSION_NAME)
#define CL_INTEL_SUBGROUP_MATRIX_MULTIPLY_ACCUMULATE_TF32_EXTENSION_NAME \
    "cl_intel_subgroup_matrix_multiply_accumulate_tf32"
#endif

constexpr size_t MMA_ITERATIONS = 8192;
constexpr size_t MMA_SETS   = 12;
constexpr size_t MMA_M      = 8;

static const char kernelString[] = R"CLC(
#define UNROLL_FACTOR 32

#define PP_CONCAT_DIRECT(X,Y) X##Y
#define PP_CONCAT(X,Y) PP_CONCAT_DIRECT(X,Y)

__attribute__((intel_reqd_sub_group_size(16)))
kernel void mma_int8(int in, global int *out)
{
#if MMA_M > 1
    typedef PP_CONCAT(short, MMA_M) ATYPE;
    typedef PP_CONCAT(int, MMA_M) CTYPE;
#else
    typedef short ATYPE
    typedef int CTYPE
#endif

    ATYPE a[MMA_SETS];
    int8 b[MMA_SETS];
    CTYPE c[MMA_SETS];

    __attribute__((opencl_unroll_hint(MMA_SETS)))
    for (int set = 0; set < MMA_SETS; set++) {
        int factor = ((15 - 1 - get_sub_group_local_id()) << 10) | (1 << 15);
        a[set] = (ATYPE)(factor);
        b[set] = (int8)(factor | (factor << 8) | (factor << 16) | (factor << 24));
        c[set] = (CTYPE)(in);
    }

    __attribute__((opencl_unroll_hint(UNROLL_FACTOR)))
    for (int iteration = 0; iteration < MMA_ITERATIONS; iteration++) {
        __attribute__((opencl_unroll_hint(MMA_SETS)))
        for (int set = 0; set < MMA_SETS; set++) {
            c[set] = intel_sub_group_i8_i8_matrix_mad_k32(a[set], b[set], c[set]);
        }
    }

    int result = 0;
    __attribute__((opencl_unroll_hint(MMA_SETS)))
    for (int set = 0; set < MMA_SETS; set++) {
#if MMA_M > 1
        result += c[set].x;
#else
        result += c[set];
#endif
    }

    out[get_global_id(0)] = result;
}

__attribute__((intel_reqd_sub_group_size(16)))
kernel void mma_fp16(float in, global float *out)
{
#if MMA_M > 1
    typedef PP_CONCAT(short, MMA_M) ATYPE;
    typedef PP_CONCAT(float, MMA_M) CTYPE;
#else
    typedef short ATYPE
    typedef float CTYPE
#endif

    ATYPE a[MMA_SETS];
    int8 b[MMA_SETS];
    CTYPE c[MMA_SETS];

    __attribute__((opencl_unroll_hint(MMA_SETS)))
    for (int set = 0; set < MMA_SETS; set++) {
        int factor = ((15 - 1 - get_sub_group_local_id()) << 10) | (1 << 15);
        a[set] = (ATYPE)(factor);
        b[set] = (int8)(factor | (factor << 16));
        c[set] = (CTYPE)(in);
    }

    __attribute__((opencl_unroll_hint(UNROLL_FACTOR)))
    for (int iteration = 0; iteration < MMA_ITERATIONS; iteration++) {
        __attribute__((opencl_unroll_hint(MMA_SETS)))
        for (int set = 0; set < MMA_SETS; set++) {
            c[set] = intel_sub_group_f16_f16_matrix_mad_k16(a[set], b[set], c[set]);
        }
    }

    float result = 0.0f;
    __attribute__((opencl_unroll_hint(MMA_SETS)))
    for (int set = 0; set < MMA_SETS; set++) {
#if MMA_M > 1
        result += c[set].x;
#else
        result += c[set];
#endif
    }

    out[get_global_id(0)] = result;
}

#if defined(cl_intel_subgroup_matrix_multiply_accumulate_tf32)
__attribute__((intel_reqd_sub_group_size(16)))
kernel void mma_tf32(float in, global float *out)
{
#if MMA_M > 1
    typedef PP_CONCAT(float, MMA_MDIV2) ATYPE;
    typedef PP_CONCAT(float, MMA_M) CTYPE;
#else
    typedef float ATYPE
    typedef float CTYPE
#endif

    ATYPE a[MMA_SETS];
    float8 b[MMA_SETS];
    CTYPE c[MMA_SETS];

    __attribute__((opencl_unroll_hint(MMA_SETS)))
    for (int set = 0; set < MMA_SETS; set++) {
        float factor = as_float(((15 - 1 - get_sub_group_local_id()) << 10) | (1 << 15));
        a[set] = (ATYPE)(factor);
        b[set] = (float8)(factor);
        c[set] = (CTYPE)(in);
    }

    __attribute__((opencl_unroll_hint(UNROLL_FACTOR)))
    for (int iteration = 0; iteration < MMA_ITERATIONS; iteration++) {
        __attribute__((opencl_unroll_hint(MMA_SETS)))
        for (int set = 0; set < MMA_SETS; set++) {
            c[set] = intel_sub_group_tf32_tf32_matrix_mad_k8(a[set], b[set], c[set]);
        }
    }

    float result = 0.0f;
    __attribute__((opencl_unroll_hint(MMA_SETS)))
    for (int set = 0; set < MMA_SETS; set++) {
#if MMA_M > 1
        result += c[set].x;
#else
        result += c[set];
#endif
    }

    out[get_global_id(0)] = result;
}
#endif
)CLC";

static size_t findMinSubGroupSize(cl::Device& device)
{
    auto s = device.getInfo<CL_DEVICE_SUB_GROUP_SIZES_INTEL>();
    auto it = std::min_element(std::begin(s), std::end(s));
    if (it != std::end(s)) {
        return *it;
    }
    return 0;
}

static float hw_time(cl::Event& event)
{
    auto ns = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
              event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    return ns / 1e9f;
}

int main(int argc, char** argv)
{
    int platformIndex = 0;
    int deviceIndex = 0;

    size_t iterations = 16;

    bool wallclock = false;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("i", "iterations", "Test Iterations", iterations, &iterations);
        op.add<popl::Switch>("", "wallclock", "Measure Wallclock Time", &wallclock);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: mmpeak [options]\n"
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

    cl::Device& device = devices[deviceIndex];
    printf("Running on device: %s (%uCUs, %uMHz)\n",
        device.getInfo<CL_DEVICE_NAME>().c_str(),
        device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(),
        device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>());
    printf("Running on drivers: %s\n",
        device.getInfo<CL_DRIVER_VERSION>().c_str());

    bool has_mma =
        checkDeviceForExtension(device, CL_INTEL_SUBGROUP_MATRIX_MULTIPLY_ACCUMULATE_EXTENSION_NAME);
    if (has_mma) {
        printf("Device supports " CL_INTEL_SUBGROUP_MATRIX_MULTIPLY_ACCUMULATE_EXTENSION_NAME ".\n");
    } else {
        printf("Device does not support " CL_INTEL_SUBGROUP_MATRIX_MULTIPLY_ACCUMULATE_EXTENSION_NAME ", exiting.\n");
        return -1;
    }

    bool has_mma_tf32 =
        checkDeviceForExtension(device, CL_INTEL_SUBGROUP_MATRIX_MULTIPLY_ACCUMULATE_TF32_EXTENSION_NAME);
    if (has_mma_tf32) {
        printf("Device supports " CL_INTEL_SUBGROUP_MATRIX_MULTIPLY_ACCUMULATE_TF32_EXTENSION_NAME ".\n");
    }

    auto minSubGroupSize = findMinSubGroupSize(device);
    if (minSubGroupSize != 16) {
        printf("This test currently requires a minimum sub-group size of 16.\n");
        printf("The device reports minimum sub-group size of %zu, exiting.\n", minSubGroupSize);
        return -1;
    }

    const size_t groupSize = minSubGroupSize;
    const size_t groupCount =
        device.getInfo<CL_DEVICE_NUM_SLICES_INTEL>() *
        device.getInfo<CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL>() *
        device.getInfo<CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL>() *
        device.getInfo<CL_DEVICE_NUM_THREADS_PER_EU_INTEL>() *
        16;
    printf("Running %zu iterations, Group count: %zu, Group size: %zu\n", iterations, groupCount, groupSize);

    cl::Context context{device};
    cl::CommandQueue commandQueue{context, device, CL_QUEUE_PROFILING_ENABLE};

    std::string buildOptions;
    buildOptions += " -DMMA_ITERATIONS=" + std::to_string(MMA_ITERATIONS);
    buildOptions += " -DMMA_SETS=" + std::to_string(MMA_SETS);
    buildOptions += " -DMMA_M=" + std::to_string(MMA_M);
    buildOptions += " -DMMA_MDIV2=" + std::to_string(MMA_M / 2);

    cl::Program program{ context, kernelString };
    program.build(buildOptions);

    cl::Buffer deviceMemDst = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        groupSize * groupCount * sizeof( cl_float ) };

    const auto run_test = [=](const std::string& typeName, size_t K) {
        const std::string kernelName = "mma_" + typeName;
        cl::Kernel kernel = cl::Kernel{ program, kernelName };

        if (typeName.find("int") != std::string::npos) {
            kernel.setArg(0, 1);
        } else {
            kernel.setArg(0, 1.0f);
        }
        kernel.setArg(1, deviceMemDst);

        commandQueue.finish();

        float best = 999.0f;
        for( size_t i = 0; i < iterations; i++ )
        {
            cl::Event event;
            auto start = std::chrono::system_clock::now();
            commandQueue.enqueueNDRangeKernel(
                kernel,
                cl::NullRange,
                cl::NDRange{groupSize * groupCount},
                cl::NDRange{groupSize},
                nullptr,
                &event);
            commandQueue.finish();
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<float> sw_time = end - start;
            auto elapsed = wallclock ? sw_time.count() : hw_time(event);
            best = std::min(best, elapsed);
        }

        auto gops =
            2.0 * K * // 2 for MAD
            groupSize * groupCount *
            MMA_ITERATIONS *
            MMA_SETS *
            MMA_M / best / 1e9;
        printf("%s: Best in %f seconds (%f gops)\n", typeName.c_str(), best, gops);
    };

    run_test("int8", 32);
    run_test("fp16", 16);
    if (has_mma_tf32) {
        run_test("tf32", 8);
    }

    return 0;
}
