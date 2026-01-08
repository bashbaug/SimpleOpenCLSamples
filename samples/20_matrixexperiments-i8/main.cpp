/*
// Copyright (c) 2024-2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <random>
#include <vector>

#include "util.hpp"

using test_clock = std::chrono::high_resolution_clock;

bool zeroData = false;
bool identityData = false;
bool fixedData = false;
bool validate = false;
bool emulate = false;
bool wallclock = false;
bool skipinit = false;
bool roundRobin = false;
int testIterations = 16;
float threshold = 0.01f;

std::string makeTestName(
    const std::string &func,
    size_t M, size_t N, size_t K)
{
    std::ostringstream ret;
    ret << func;
    ret << " (M=" << M << ", N=" << N << ", K=" << K << ")";
    return ret.str();
}

std::string makeTestName(
    const std::string &func,
    int tM, int tN,
    size_t M, size_t N, size_t K)
{
    std::ostringstream ret;
    ret << func;
    ret << "<tM:" << tM << ", tN:" << tN << ">";
    ret << " (M=" << M << ", N=" << N << ", K=" << K << ")";
    return ret.str();
}

std::string makeTestName(
    const std::string &func,
    int tM, int tN,
    int MM, int NN,
    size_t M, size_t N, size_t K)
{
    std::ostringstream ret;
    ret << func;
    ret << "<tM:" << tM << "x" << MM << ", tN:" << tN << "x" << NN << ">";
    ret << " (M=" << M << ", N=" << N << ", K=" << K << ")";
    return ret.str();
}

static size_t findMinSubGroupSize(cl::Device& device)
{
    auto s = device.getInfo<CL_DEVICE_SUB_GROUP_SIZES_INTEL>();
    auto it = std::min_element(std::begin(s), std::end(s));
    if (it != std::end(s)) {
        return *it;
    }
    return 0;
}

static void setRoundRobin(cl::Kernel& kernel)
{
    constexpr cl_kernel_exec_info CL_KERNEL_EXEC_INFO_THREAD_ARBITRATION_POLICY_INTEL = 0x10025;
    constexpr cl_uint CL_KERNEL_EXEC_INFO_THREAD_ARBITRATION_POLICY_ROUND_ROBIN_INTEL = 0x10023;
    const cl_uint policy = CL_KERNEL_EXEC_INFO_THREAD_ARBITRATION_POLICY_ROUND_ROBIN_INTEL;
    clSetKernelExecInfo(
        kernel(),
        CL_KERNEL_EXEC_INFO_THREAD_ARBITRATION_POLICY_INTEL,
        sizeof(policy),
        &policy);
}

template <typename T>
static void fill_matrix(std::vector<T>& M, size_t numRows, size_t numCols)
{
    if (zeroData) {
        std::generate(std::begin(M), std::end(M), [&]{ return 0; });
    }
    else if (identityData) {
        std::generate(std::begin(M), std::end(M), [&]{ return 1; });
    } else if (fixedData) {
        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++) {
                M[r * numCols + c] = static_cast<T>(r + c);
            }
        }
    } else {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<int> dist(-64, 64);
        std::generate(std::begin(M), std::end(M), [&]{ return dist(rng); });
    }
}

template <typename T>
static void vnni_matrix(
    std::vector<T> &dst, const std::vector<T> &src,
    size_t numRows, size_t numCols, size_t factor)
{
    for (size_t r = 0; r < numRows / factor; r++) {
        for (size_t c = 0; c < numCols; c++) {
            for (size_t k = 0; k < factor; k++) {
                dst[r * numCols * factor + c * factor + k] =
                    src[(r * factor + k) * numCols + c];
            }
        }
    }
}

template <typename DstT, typename SrcT>
static void compute_reference(
    std::vector<DstT>& C,
    const std::vector<SrcT>& A, const std::vector<SrcT>& B,
    size_t M, size_t N, size_t K)
{
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            DstT sum = 0;
            for (size_t k = 0; k < K; k++) {
                sum = A[m * K + k] * B[k * N + n] + sum;
            }
            C[m * N + n] = sum;
        }
    }
}

template <typename T>
void check_results(
    size_t M,
    size_t N,
    const std::vector<T>& C,
    const std::vector<T>& C_ref)
{
    float err = 0.f;
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            auto index = m * N + n;
            if (C[index] != C_ref[index]) {
                std::cerr << "Error at m = " << m << ", n = " << n
                          << ": Wanted "
                          << C_ref[index] << ", got " << C[index] << std::endl;
                return;
            }
        }
    }
}

static float hw_time(cl::Event& event)
{
    auto ns = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
              event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    return ns / 1e9f;
}

static void i8_naive(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
    size_t M, size_t N, size_t K,
    const std::vector<int>& C_ref)
{
    printf("%80s: ", makeTestName(__FUNCTION__, M, N, K).c_str()); fflush(stdout);

    cl::Kernel kernel{program, "i8_naive"};
    if (kernel() == nullptr) {
        printf("unsupported.\n");
    } else {
        kernel.setArg(0, C);
        kernel.setArg(1, A);
        kernel.setArg(2, B);
        kernel.setArg(3, static_cast<cl_int>(K));

        if (!skipinit) {
            queue.enqueueFillBuffer(C, 0, 0, C_ref.size() * sizeof(C_ref[0]));
        }

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            cl::Event event;
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                cl::NDRange{N, M}, cl::NullRange, nullptr, &event);
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> sw_time = end - start;
            auto elapsed = wallclock ? sw_time.count() : hw_time(event);
            best = std::min(best, elapsed);
        }
        auto gops = 2.0 * M * N * K / best / 1e9;
        printf("Best in %f seconds (%f gops)\n", best, gops);

        if (validate) {
            printf("Checking results... "); fflush(stdout);
            std::vector<int> C_check(C_ref.size());
            queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(C_check[0]), C_check.data());
            check_results(M, N, C_check, C_ref);
            printf(" done!\n");
        }
    }
}

template<int tM, int tN>
static void i8_dpas_rowmajor(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
    size_t M, size_t N, size_t K,
    const std::vector<int>& C_ref)
{
    printf("%80s: ", makeTestName(__FUNCTION__, tM, tN, M, N, K).c_str()); fflush(stdout);

    std::string kernelName = "i8_dpas_rowmajor";
    kernelName += "_m" + std::to_string(tM);
    kernelName += "_n" + std::to_string(tN);
    cl::Kernel kernel{program, kernelName.c_str()};
    if (kernel() == nullptr) {
        printf("unsupported.\n");
    } else {
        kernel.setArg(0, C);
        kernel.setArg(1, A);
        kernel.setArg(2, B);
        kernel.setArg(3, static_cast<cl_int>(K));

        if (!skipinit) {
            queue.enqueueFillBuffer(C, 0, 0, C_ref.size() * sizeof(C_ref[0]));
        }

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            cl::Event event;
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                cl::NDRange{N, M/tM}, cl::NullRange, nullptr, &event);
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> sw_time = end - start;
            auto elapsed = wallclock ? sw_time.count() : hw_time(event);
            best = std::min(best, elapsed);
        }
        auto gops = 2.0 * M * N * K / best / 1e9;
        printf("Best in %f seconds (%f gops)\n", best, gops);

        if (validate) {
            printf("Checking results... "); fflush(stdout);
            std::vector<int> C_check(C_ref.size());
            queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(C_check[0]), C_check.data());
            check_results(M, N, C_check, C_ref);
            printf(" done!\n");
        }
    }
}

template<int tM, int tN>
static void i8_dpas_vnni(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
    size_t M, size_t N, size_t K,
    const std::vector<int>& C_ref)
{
    printf("%80s: ", makeTestName(__FUNCTION__, tM, tN, M, N, K).c_str()); fflush(stdout);

    std::string kernelName = "i8_dpas_vnni";
    kernelName += "_m" + std::to_string(tM);
    kernelName += "_n" + std::to_string(tN);
    cl::Kernel kernel{program, kernelName.c_str()};
    if (kernel() == nullptr) {
        printf("unsupported.\n");
    } else {
        kernel.setArg(0, C);
        kernel.setArg(1, A);
        kernel.setArg(2, B);
        kernel.setArg(3, static_cast<cl_int>(K));

        if (!skipinit) {
            queue.enqueueFillBuffer(C, 0, 0, C_ref.size() * sizeof(C_ref[0]));
        }

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            cl::Event event;
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                cl::NDRange{N, M/tM}, cl::NullRange, nullptr, &event);
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> sw_time = end - start;
            auto elapsed = wallclock ? sw_time.count() : hw_time(event);
            best = std::min(best, elapsed);
        }
        auto gops = 2.0 * M * N * K / best / 1e9;
        printf("Best in %f seconds (%f gops)\n", best, gops);

        if (validate) {
            printf("Checking results... "); fflush(stdout);
            std::vector<int> C_check(C_ref.size());
            queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(C_check[0]), C_check.data());
            check_results(M, N, C_check, C_ref);
            printf(" done!\n");
        }
    }
}

template<int tM, int tN>
static void i8_dpas_blockread_rowmajor(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
    size_t M, size_t N, size_t K,
    const std::vector<int>& C_ref)
{
    printf("%80s: ", makeTestName(__FUNCTION__, tM, tN, M, N, K).c_str()); fflush(stdout);

    std::string kernelName = "i8_dpas_blockread_rowmajor";
    kernelName += "_m" + std::to_string(tM);
    kernelName += "_n" + std::to_string(tN);
    cl::Kernel kernel{program, kernelName.c_str()};
    if (kernel() == nullptr) {
        printf("unsupported.\n");
    } else if (K < 64 || N < 64) {
        printf("matrix pitch for block reads must be >= 64 bytes.\n");
    } else {
        kernel.setArg(0, C);
        kernel.setArg(1, A);
        kernel.setArg(2, B);
        kernel.setArg(3, static_cast<cl_int>(K));
        if (roundRobin) {
            setRoundRobin(kernel);
        }

        if (!skipinit) {
            queue.enqueueFillBuffer(C, 0, 0, C_ref.size() * sizeof(C_ref[0]));
        }

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            cl::Event event;
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                cl::NDRange{N, M/tM}, cl::NullRange, nullptr, &event);
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> sw_time = end - start;
            auto elapsed = wallclock ? sw_time.count() : hw_time(event);
            best = std::min(best, elapsed);
        }
        auto gops = 2.0 * M * N * K / best / 1e9;
        printf("Best in %f seconds (%f gops)\n", best, gops);

        if (validate) {
            printf("Checking results... "); fflush(stdout);
            std::vector<int> C_check(C_ref.size());
            queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(C_check[0]), C_check.data());
            check_results(M, N, C_check, C_ref);
            printf(" done!\n");
        }
    }
}

template<int tM, int tN>
static void i8_dpas_blockread_vnni(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
    size_t M, size_t N, size_t K,
    const std::vector<int>& C_ref)
{
    printf("%80s: ", makeTestName(__FUNCTION__, tM, tN, M, N, K).c_str()); fflush(stdout);

    std::string kernelName = "i8_dpas_blockread_vnni";
    kernelName += "_m" + std::to_string(tM);
    kernelName += "_n" + std::to_string(tN);
    cl::Kernel kernel{program, kernelName.c_str()};
    if (kernel() == nullptr) {
        printf("unsupported.\n");
    } else if (K < 64 || N < 64/4) {
        printf("matrix pitch for block reads must be >= 64 bytes.\n");
    } else {
        kernel.setArg(0, C);
        kernel.setArg(1, A);
        kernel.setArg(2, B);
        kernel.setArg(3, static_cast<cl_int>(K));
        if (roundRobin) {
            setRoundRobin(kernel);
        }

        if (!skipinit) {
            queue.enqueueFillBuffer(C, 0, 0, C_ref.size() * sizeof(C_ref[0]));
        }

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            cl::Event event;
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                cl::NDRange{N, M/tM}, cl::NullRange, nullptr, &event);
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> sw_time = end - start;
            auto elapsed = wallclock ? sw_time.count() : hw_time(event);
            best = std::min(best, elapsed);
        }
        auto gops = 2.0 * M * N * K / best / 1e9;
        printf("Best in %f seconds (%f gops)\n", best, gops);

        if (validate) {
            printf("Checking results... "); fflush(stdout);
            std::vector<int> C_check(C_ref.size());
            queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(C_check[0]), C_check.data());
            check_results(M, N, C_check, C_ref);
            printf(" done!\n");
        }
    }
}

int main(int argc, char** argv)
{
    int platformIndex = 0;
    int deviceIndex = 0;

    std::string fileName("matrix_kernels_i8.cl");
    std::string buildOptions;
    size_t matrixSize = 512;

    size_t mask = ~0;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<std::string>>("", "file", "Kernel File Name", fileName, &fileName);
        op.add<popl::Value<std::string>>("", "options", "Program Build Options", buildOptions, &buildOptions);
        op.add<popl::Value<size_t>>("m", "matrixsize", "Matrix Size", matrixSize, &matrixSize);
        op.add<popl::Value<int>>("i", "iterations", "Test Iterations", testIterations, &testIterations);
        op.add<popl::Switch>("", "validate", "Validate Results", &validate);
        op.add<popl::Switch>("", "zero", "Use Zero Data", &zeroData);
        op.add<popl::Switch>("", "identity", "Use Identity Data", &identityData);
        op.add<popl::Switch>("", "fixed", "Use Fixed Data", &fixedData);
        op.add<popl::Switch>("", "emulate", "Unconditionally Emulate dpas", &emulate);
        op.add<popl::Switch>("", "wallclock", "Measure Wallclock Time", &wallclock);
        op.add<popl::Switch>("", "skipinit", "Do Not Initialize Buffers", &skipinit);
        op.add<popl::Switch>("", "roundrobin", "Use Round Robin Scheduling", &roundRobin);
        op.add<popl::Value<float>>("", "threshold", "Local Error Threshold", threshold, &threshold);
        op.add<popl::Value<size_t>, popl::Attribute::advanced>("", "mask", "Test Mask", mask, &mask);
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: matrixexperiments-i8 [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (!checkPlatformIndex(platforms, platformIndex)) {
        return -1;
    }

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (deviceIndex >= devices.size()) {
        printf("Requested device index is %d, but only %zu devices were found.\n",
            deviceIndex, devices.size());
    }

    cl::Device& device = devices[deviceIndex];
    printf("Running on device: %s (%uCUs, %uMHz)\n",
        device.getInfo<CL_DEVICE_NAME>().c_str(),
        device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(),
        device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>());
    printf("Running on drivers: %s\n",
        device.getInfo<CL_DRIVER_VERSION>().c_str());

    auto minSubGroupSize = findMinSubGroupSize(device);

    bool has_simd8 = minSubGroupSize == 8;
    bool emulate_tN8 = true;
    bool emulate_tN16 = true;
    if (!emulate && checkDeviceForExtension(device, "cl_intel_subgroup_matrix_multiply_accumulate")) {
        printf("Found support for cl_intel_subgroup_matrix_multiply_accumulate, min sub-group size is: %zu\n", minSubGroupSize);
        switch(minSubGroupSize) {
            case 8: emulate_tN8 = false; break;
            case 16: emulate_tN16 = false; break;
            default: break;
        }
    }

    buildOptions += " -DHAS_SIMD8=" + std::to_string(has_simd8);
    buildOptions += " -DEMULATE_tN8=" + std::to_string(emulate_tN8);
    buildOptions += " -DEMULATE_tN16=" + std::to_string(emulate_tN16);

    printf("Config:\n");
    printf("\tTest Iterations: %d\n", testIterations);
    printf("\tValidating data?: %s\n", validate ? "true" : "false");
    printf("\tFixed data?: %s\n", fixedData ? "true" : "false");
    printf("\tWallclock time?: %s\n", wallclock ? "true" : "false");
    printf("\tEmulate dpas for tN=8?: %s\n", emulate_tN8 ? "true" : "false");
    printf("\tEmulate dpas for tN=16?: %s\n", emulate_tN16 ? "true" : "false");

    cl::Context context{device};
    cl::CommandQueue queue{context, device, CL_QUEUE_PROFILING_ENABLE};

    printf("Reading program source from file: %s\n", fileName.c_str() );
    std::string kernelString = readStringFromFile(fileName.c_str());

    printf("Building program with build options: %s\n",
        buildOptions.empty() ? "(none)" : buildOptions.c_str() );
    cl::Program program{ context, kernelString };
    program.build(buildOptions.c_str());
    for( auto& device : program.getInfo<CL_PROGRAM_DEVICES>() )
    {
        printf("Program build log for device %s:\n",
            device.getInfo<CL_DEVICE_NAME>().c_str() );
        printf("%s\n",
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str() );
    }

    const auto M = matrixSize;
    const auto N = matrixSize;
    const auto K = matrixSize;

    std::vector<int8_t> A_vec(M * K);
    std::vector<int8_t> B_vec(K * N);
    std::vector<int8_t> Bvnni_vec(K * N);

    std::vector<int> C_ref(M * N);

    printf("Initializing source matrices...\n");
    fill_matrix(A_vec, M, K);
    fill_matrix(B_vec, K, N);

    vnni_matrix(Bvnni_vec, B_vec, K, N, 4);

    if (validate) {
        printf("Computing reference...\n");
        compute_reference(C_ref, A_vec, B_vec, M, N, K);
    }

    printf("Creating source buffers...\n");
    cl::Buffer A{context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, A_vec.size() * sizeof(A_vec[0]), A_vec.data()};
    cl::Buffer B{context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, B_vec.size() * sizeof(B_vec[0]), B_vec.data()};
    cl::Buffer Bvnni{context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Bvnni_vec.size() * sizeof(Bvnni_vec[0]), Bvnni_vec.data()};
    cl::Buffer C{context, CL_MEM_WRITE_ONLY, C_ref.size() * sizeof(C_ref[0])};

    printf("Running tests...\n");

    if (mask & 0x1) {
        i8_naive(context, program, queue, C, A, B, M, N, K, C_ref);
    }

    if (mask & 0x2) {
        i8_dpas_rowmajor<1, 8>(context, program, queue, C, A, B, M, N, K, C_ref);
        i8_dpas_rowmajor<2, 8>(context, program, queue, C, A, B, M, N, K, C_ref);
        i8_dpas_rowmajor<4, 8>(context, program, queue, C, A, B, M, N, K, C_ref);
        i8_dpas_rowmajor<8, 8>(context, program, queue, C, A, B, M, N, K, C_ref);
    }

    if (mask & 0x20) {
        i8_dpas_rowmajor<1, 16>(context, program, queue, C, A, B, M, N, K, C_ref);
        i8_dpas_rowmajor<2, 16>(context, program, queue, C, A, B, M, N, K, C_ref);
        i8_dpas_rowmajor<4, 16>(context, program, queue, C, A, B, M, N, K, C_ref);
        i8_dpas_rowmajor<8, 16>(context, program, queue, C, A, B, M, N, K, C_ref);
    }

    if (mask & 0x80) {
        i8_dpas_vnni<1, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
        i8_dpas_vnni<2, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
        i8_dpas_vnni<4, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
        i8_dpas_vnni<8, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    }

    if (mask & 0x200) {
        i8_dpas_blockread_rowmajor<1, 16>(context, program, queue, C, A, B, M, N, K, C_ref);
        i8_dpas_blockread_rowmajor<2, 16>(context, program, queue, C, A, B, M, N, K, C_ref);
        i8_dpas_blockread_rowmajor<4, 16>(context, program, queue, C, A, B, M, N, K, C_ref);
        i8_dpas_blockread_rowmajor<8, 16>(context, program, queue, C, A, B, M, N, K, C_ref);
    }

    if (mask & 0x800) {
        i8_dpas_blockread_vnni<1, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
        i8_dpas_blockread_vnni<2, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
        i8_dpas_blockread_vnni<4, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
        i8_dpas_blockread_vnni<8, 16>(context, program, queue, C, A, Bvnni, M, N, K, C_ref);
    }

    printf("Done.\n");

    return 0;
}
