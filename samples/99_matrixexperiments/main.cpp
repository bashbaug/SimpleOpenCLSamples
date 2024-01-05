/*
// Copyright (c) 2019-2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <algorithm>
#include <chrono>
#include <string>
#include <random>

#include "bfloat16.hpp"
#include "util.hpp"

using test_clock = std::chrono::high_resolution_clock;

bool fixedData = false;
bool validate = false;
int testIterations = 16;
float threshold = 0.01f;

template <typename T>
static void fill_matrix(std::vector<T>& M, size_t numRows, size_t numCols)
{
    if (fixedData) {
        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++) {
                M[r * numCols + c] = r + c;
            }
        }
    } else {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<float> dist(-1.0, 1.0);
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
                sum = std::fma(static_cast<DstT>(A[m * K + k]),
                               static_cast<DstT>(B[k * N + n]), sum);
            }
            C[m * N + n] = sum;
        }
    }
}

template <typename T>
int check_results(const std::vector<T>& C,
                  const std::vector<T>& C_ref)
{
    float err = 0.f;
    for (int i = 0; i < C.size(); ++i) {
        float localErr = std::fabs(C[i] - C_ref[i]) /
                         std::max(std::fabs(C[i]),
                                  std::fabs(C_ref[i]));
        err = std::max(localErr, err);
        if (localErr >= threshold) {
            std::cerr << "Error at index " << i << " (local error " << localErr
                      << "): Wanted " << C_ref[i] << ", got " << C[i]
                      << std::endl;
            break;
        }
    }

    return err < 0.001f;
}

template<typename T>
static void go_naive(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
    size_t M, size_t N, size_t K,
    const std::vector<T>& C_ref)
{
    printf("%40s (M=%zu, N = %zu, K = %zu): ", __FUNCTION__, M, N, K); fflush(stdout);

    cl::Kernel kernel{program, "bfloat16_naive"};
    kernel.setArg(0, C);
    kernel.setArg(1, A);
    kernel.setArg(2, B);
    kernel.setArg(3, static_cast<cl_int>(K));

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{N, M});
        queue.finish();
        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    auto gops = 2.0 * M * N * K / best / 1e9;
    printf("Finished in %f seconds (%f gops)\n", best, gops);

    if (validate) {
        printf("Checking results... "); fflush(stdout);
        std::vector<T> C_check(C_ref.size());
        queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(T), C_check.data());
        check_results(C_check, C_ref);
        printf(" done!\n");
    }
}

template<typename T>
static void go_dpas_rowmajor_m1(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
    size_t M, size_t N, size_t K,
    const std::vector<T>& C_ref)
{
    printf("%40s (M=%zu, N = %zu, K = %zu): ", __FUNCTION__, M, N, K); fflush(stdout);

    cl::Kernel kernel{program, "bfloat16_dpas_rowmajor_m1"};
    if (kernel()) {
        kernel.setArg(0, C);
        kernel.setArg(1, A);
        kernel.setArg(2, B);
        kernel.setArg(3, static_cast<cl_int>(K));

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{N, M});
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            best = std::min(best, elapsed_seconds.count());
        }
        auto gops = 2.0 * M * N * K / best / 1e9;
        printf("Finished in %f seconds (%f gops)\n", best, gops);

        if (validate) {
            printf("Checking results... "); fflush(stdout);
            std::vector<T> C_check(C_ref.size());
            queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(T), C_check.data());
            check_results(C_check, C_ref);
            printf(" done!\n");
        }
    } else {
        printf("unsupported.\n");
    }
}

template<typename T>
static void go_dpas_rowmajor_m2(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
    size_t M, size_t N, size_t K,
    const std::vector<T>& C_ref)
{
    printf("%40s (M=%zu, N = %zu, K = %zu): ", __FUNCTION__, M, N, K); fflush(stdout);

    cl::Kernel kernel{program, "bfloat16_dpas_rowmajor_m2"};
    if (kernel()) {
        kernel.setArg(0, C);
        kernel.setArg(1, A);
        kernel.setArg(2, B);
        kernel.setArg(3, static_cast<cl_int>(K));

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{N, M/2});
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            best = std::min(best, elapsed_seconds.count());
        }
        auto gops = 2.0 * M * N * K / best / 1e9;
        printf("Finished in %f seconds (%f gops)\n", best, gops);

        if (validate) {
            printf("Checking results... "); fflush(stdout);
            std::vector<T> C_check(C_ref.size());
            queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(T), C_check.data());
            check_results(C_check, C_ref);
            printf(" done!\n");
        }
    } else {
        printf("unsupported.\n");
    }
}

template<typename T>
static void go_dpas_rowmajor_m4(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
    size_t M, size_t N, size_t K,
    const std::vector<T>& C_ref)
{
    printf("%40s (M=%zu, N = %zu, K = %zu): ", __FUNCTION__, M, N, K); fflush(stdout);

    cl::Kernel kernel{program, "bfloat16_dpas_rowmajor_m4"};
    if (kernel()) {
        kernel.setArg(0, C);
        kernel.setArg(1, A);
        kernel.setArg(2, B);
        kernel.setArg(3, static_cast<cl_int>(K));

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{N, M/4});
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            best = std::min(best, elapsed_seconds.count());
        }
        auto gops = 2.0 * M * N * K / best / 1e9;
        printf("Finished in %f seconds (%f gops)\n", best, gops);

        if (validate) {
            printf("Checking results... "); fflush(stdout);
            std::vector<T> C_check(C_ref.size());
            queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(T), C_check.data());
            check_results(C_check, C_ref);
            printf(" done!\n");
        }
    } else {
        printf("unsupported.\n");
    }
}

template<typename T>
static void go_dpas_rowmajor_m8(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
    size_t M, size_t N, size_t K,
    const std::vector<T>& C_ref)
{
    printf("%40s (M=%zu, N = %zu, K = %zu): ", __FUNCTION__, M, N, K); fflush(stdout);

    cl::Kernel kernel{program, "bfloat16_dpas_rowmajor_m8"};
    if (kernel()) {
        kernel.setArg(0, C);
        kernel.setArg(1, A);
        kernel.setArg(2, B);
        kernel.setArg(3, static_cast<cl_int>(K));

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{N, M/8});
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            best = std::min(best, elapsed_seconds.count());
        }
        auto gops = 2.0 * M * N * K / best / 1e9;
        printf("Finished in %f seconds (%f gops)\n", best, gops);

        if (validate) {
            printf("Checking results... "); fflush(stdout);
            std::vector<T> C_check(C_ref.size());
            queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(T), C_check.data());
            check_results(C_check, C_ref);
            printf(" done!\n");
        }
    } else {
        printf("unsupported.\n");
    }
}

template<typename T>
static void go_dpas_vnni_m1(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
    size_t M, size_t N, size_t K,
    const std::vector<T>& C_ref)
{
    printf("%40s (M=%zu, N = %zu, K = %zu): ", __FUNCTION__, M, N, K); fflush(stdout);

    cl::Kernel kernel{program, "bfloat16_dpas_vnni_m1"};
    if (kernel()) {
        kernel.setArg(0, C);
        kernel.setArg(1, A);
        kernel.setArg(2, B);
        kernel.setArg(3, static_cast<cl_int>(K));

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{N, M});
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            best = std::min(best, elapsed_seconds.count());
        }
        auto gops = 2.0 * M * N * K / best / 1e9;
        printf("Finished in %f seconds (%f gops)\n", best, gops);

        if (validate) {
            printf("Checking results... "); fflush(stdout);
            std::vector<T> C_check(C_ref.size());
            queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(T), C_check.data());
            check_results(C_check, C_ref);
            printf(" done!\n");
        }
    } else {
        printf("unsupported.\n");
    }
}

template<typename T>
static void go_dpas_vnni_m2(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
    size_t M, size_t N, size_t K,
    const std::vector<T>& C_ref)
{
    printf("%40s (M=%zu, N = %zu, K = %zu): ", __FUNCTION__, M, N, K); fflush(stdout);

    cl::Kernel kernel{program, "bfloat16_dpas_vnni_m2"};
    if (kernel()) {
        kernel.setArg(0, C);
        kernel.setArg(1, A);
        kernel.setArg(2, B);
        kernel.setArg(3, static_cast<cl_int>(K));

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{N, M/2});
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            best = std::min(best, elapsed_seconds.count());
        }
        auto gops = 2.0 * M * N * K / best / 1e9;
        printf("Finished in %f seconds (%f gops)\n", best, gops);

        if (validate) {
            printf("Checking results... "); fflush(stdout);
            std::vector<T> C_check(C_ref.size());
            queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(T), C_check.data());
            check_results(C_check, C_ref);
            printf(" done!\n");
        }
    } else {
        printf("unsupported.\n");
    }
}

template<typename T>
static void go_dpas_vnni_m4(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
    size_t M, size_t N, size_t K,
    const std::vector<T>& C_ref)
{
    printf("%40s (M=%zu, N = %zu, K = %zu): ", __FUNCTION__, M, N, K); fflush(stdout);

    cl::Kernel kernel{program, "bfloat16_dpas_vnni_m4"};
    if (kernel()) {
        kernel.setArg(0, C);
        kernel.setArg(1, A);
        kernel.setArg(2, B);
        kernel.setArg(3, static_cast<cl_int>(K));

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{N, M/4});
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            best = std::min(best, elapsed_seconds.count());
        }
        auto gops = 2.0 * M * N * K / best / 1e9;
        printf("Finished in %f seconds (%f gops)\n", best, gops);

        if (validate) {
            printf("Checking results... "); fflush(stdout);
            std::vector<T> C_check(C_ref.size());
            queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(T), C_check.data());
            check_results(C_check, C_ref);
            printf(" done!\n");
        }
    } else {
        printf("unsupported.\n");
    }
}

template<typename T>
static void go_dpas_vnni_m8(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& C, cl::Buffer& A, cl::Buffer& B,
    size_t M, size_t N, size_t K,
    const std::vector<T>& C_ref)
{
    printf("%40s (M=%zu, N = %zu, K = %zu): ", __FUNCTION__, M, N, K); fflush(stdout);

    cl::Kernel kernel{program, "bfloat16_dpas_vnni_m8"};
    if (kernel()) {
        kernel.setArg(0, C);
        kernel.setArg(1, A);
        kernel.setArg(2, B);
        kernel.setArg(3, static_cast<cl_int>(K));

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{N, M/8});
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            best = std::min(best, elapsed_seconds.count());
        }
        auto gops = 2.0 * M * N * K / best / 1e9;
        printf("Finished in %f seconds (%f gops)\n", best, gops);

        if (validate) {
            printf("Checking results... "); fflush(stdout);
            std::vector<T> C_check(C_ref.size());
            queue.enqueueReadBuffer(C, CL_TRUE, 0, C_check.size() * sizeof(T), C_check.data());
            check_results(C_check, C_ref);
            printf(" done!\n");
        }
    } else {
        printf("unsupported.\n");
    }
}

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    std::string fileName("matrix_kernels.cl");
    std::string buildOptions;
    size_t matrixSize = 512;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<std::string>>("", "file", "Kernel File Name", fileName, &fileName);
        op.add<popl::Value<std::string>>("", "options", "Program Build Options", buildOptions, &buildOptions);
        op.add<popl::Value<size_t>>("m", "matrixsize", "Matrix Size", matrixSize, &matrixSize);
        op.add<popl::Value<int>>("i", "iterations", "Test Iterations", testIterations, &testIterations);
        op.add<popl::Switch>("", "validate", "Validate Results", &validate);
        op.add<popl::Switch>("", "fixed", "Use Fixed Data", &fixedData);
        op.add<popl::Value<float>>("", "threshold", "Local Error Threshold", threshold, &threshold);
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: matrixexperiments [options]\n"
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
    printf("Running on device: %s\n",
        device.getInfo<CL_DEVICE_NAME>().c_str() );

    printf("Config:\n");
    printf("\tTest Iterations: %d\n", testIterations);
    printf("\tValidating data?: %s\n", validate ? "true" : "false");
    printf("\tFixed data?: %s\n", fixedData ? "true" : "false");

    cl::Context context{device};
    cl::CommandQueue queue{context, device};

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

    std::vector<bfloat16> A(matrixSize * matrixSize);
    std::vector<bfloat16> B(matrixSize * matrixSize);
    std::vector<bfloat16> B_vnni(matrixSize * matrixSize);

    std::vector<float> C(matrixSize * matrixSize);
    std::vector<float> C_ref(matrixSize * matrixSize);

    printf("Initializing source matrices...\n");
    fill_matrix(A, matrixSize, matrixSize);
    fill_matrix(B, matrixSize, matrixSize);

    vnni_matrix(B_vnni, B, matrixSize, matrixSize, 2);

    if (validate) {
        printf("Computing reference...\n");
        compute_reference(C_ref, A, B, matrixSize, matrixSize, matrixSize);
    }

    printf("Creating source buffers...\n");
    cl::Buffer Abuf{context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, A.size() * sizeof(A[0]), A.data()};
    cl::Buffer Bbuf{context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, B.size() * sizeof(B[0]), B.data()};
    cl::Buffer Bbuf_vnni{context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, B_vnni.size() * sizeof(B_vnni[0]), B_vnni.data()};
    cl::Buffer Cbuf{context, CL_MEM_WRITE_ONLY, C.size() * sizeof(C[0])};

    printf("Running tests...\n");

    go_naive(context, program, queue, Cbuf, Abuf, Bbuf, matrixSize, matrixSize, matrixSize, C_ref);
    go_dpas_rowmajor_m1(context, program, queue, Cbuf, Abuf, Bbuf, matrixSize, matrixSize, matrixSize, C_ref);
    go_dpas_rowmajor_m2(context, program, queue, Cbuf, Abuf, Bbuf, matrixSize, matrixSize, matrixSize, C_ref);
    go_dpas_rowmajor_m4(context, program, queue, Cbuf, Abuf, Bbuf, matrixSize, matrixSize, matrixSize, C_ref);
    go_dpas_rowmajor_m8(context, program, queue, Cbuf, Abuf, Bbuf, matrixSize, matrixSize, matrixSize, C_ref);

    go_dpas_vnni_m1(context, program, queue, Cbuf, Abuf, Bbuf_vnni, matrixSize, matrixSize, matrixSize, C_ref);
    go_dpas_vnni_m2(context, program, queue, Cbuf, Abuf, Bbuf_vnni, matrixSize, matrixSize, matrixSize, C_ref);
    go_dpas_vnni_m4(context, program, queue, Cbuf, Abuf, Bbuf_vnni, matrixSize, matrixSize, matrixSize, C_ref);
    go_dpas_vnni_m8(context, program, queue, Cbuf, Abuf, Bbuf_vnni, matrixSize, matrixSize, matrixSize, C_ref);

    printf("Done.\n");

    return 0;
}
