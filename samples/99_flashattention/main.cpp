/*
// Copyright (c) 2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <sstream>
#include <string>
#include <random>
#include <vector>

#include "util.hpp"

using test_clock = std::chrono::high_resolution_clock;

constexpr int B = 8;
constexpr int T = 1024;
constexpr int C = 768;
constexpr int NH = 12;

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
static void fill_input(std::vector<T>& M, size_t count)
{
    if (zeroData) {
        std::generate(std::begin(M), std::end(M), [&]{ return 0.0f; });
    }
    else if (identityData) {
        std::generate(std::begin(M), std::end(M), [&]{ return 1.0f; });
    } else if (fixedData) {
        for (size_t i = 0; i < count; i++) {
            M[i] = static_cast<float>(i);
        }
    } else {
        //std::random_device dev;
        //std::mt19937 rng(dev());
        std::default_random_engine rng;
        std::uniform_real_distribution<float> dist(0.0, 1.0);
        std::generate(std::begin(M), std::end(M), [&]{ return dist(rng); });
    }
}

// ----------------------------------------------------------------------------
// CPU code reference - adapted from llm.c
template <typename TT>
void attention_forward_cpu(
    std::vector<TT>& out,
    std::vector<TT>& preatt,
    std::vector<TT>& att,
    const std::vector<TT>& inp)
{
    // input is (B, T, 3C) Q,K,V
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    TT scale = 1.0 / std::sqrt(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const TT* query_t = inp.data() + b * T * C3 + t * C3 + h * hs;
                TT* preatt_bth = preatt.data() + b*NH*T*T + h*T*T + t*T;
                TT* att_bth = att.data() + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                TT maxval = -10000.0; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    const TT* key_t2 = inp.data() + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    TT val = 0.0;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }
                // pad with -INFINITY outside of autoregressive region for debugging comparisons
                for (int t2 = t+1; t2 < T; t2++) {
                    preatt_bth[t2] = -INFINITY;
                }

                // pass 2: calculate the exp and keep track of sum
                TT expsum = 0.0;
                for (int t2 = 0; t2 <= t; t2++) {
                    TT expv = std::exp(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                TT expsum_inv = expsum == 0.0 ? 0.0 : 1.0 / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                TT* out_bth = out.data() + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0; }
                for (int t2 = 0; t2 <= t; t2++) {
                    const TT* value_t2 = inp.data() + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    TT att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

template <typename TT>
void check_results(
    const std::vector<TT>& check,
    const std::vector<TT>& reference,
    TT tolerance = 1e-4)
{
    if (check.size() != reference.size()) {
        printf("Size mismatch?  %zu vs %zu\n", check.size(), reference.size());
    }
    for (size_t i = 0; i < check.size(); i++) {
        // if (i < 5) {
        //     printf("%f %f\n", reference[i], check[i]);
        // }
        if (fabs(reference[i] - check[i]) > tolerance) {
            printf("Mismatch at %zu: %f vs %f\n", i, reference[i], check[i]);
            return;
        }
    }
}

template<class TT>
TT ceil_div(TT dividend, TT divisor) {
    return (dividend + divisor - 1) / divisor;
}

static float hw_time(cl::Event& event)
{
    auto ns = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
              event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    return ns / 1e9f;
}

// This implements the naive algorithm.
// It has separate kernels for each of the three steps.
static void naive_attention_forward(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& out,
    cl::Buffer& preatt, cl::Buffer& att, cl::Buffer& inp,
    size_t wgSize,
    const std::vector<float>& out_ref,
    const std::vector<float>& preatt_ref,
    const std::vector<float>& att_ref)
{
    printf("%80s: ", __FUNCTION__); fflush(stdout);

    cl::Kernel naive_query_key{program, "naive_query_key"};
    cl::Kernel naive_softmax{program, "naive_softmax"};
    cl::Kernel naive_value{program, "naive_value"};
    if (naive_query_key() == nullptr ||
        naive_softmax() == nullptr ||
        naive_value() == nullptr) {
        printf("unsupported.\n");
    } else {
        size_t native_query_key_gws = ceil_div<size_t>(B * NH * T * T, wgSize) * wgSize;
        naive_query_key.setArg(0, preatt);
        naive_query_key.setArg(1, inp);

        size_t naive_softmax_gws = ceil_div<size_t>(B * T * NH, wgSize) * wgSize;
        naive_softmax.setArg(0, att);
        naive_softmax.setArg(1, preatt);

        size_t naive_value_gws = ceil_div<size_t>(B * T * NH, wgSize) * wgSize;
        naive_value.setArg(0, out);
        naive_value.setArg(1, att);
        naive_value.setArg(2, inp);

        if (!skipinit) {
            queue.enqueueFillBuffer(out, 0, 0, out_ref.size() * sizeof(out_ref[0]));
        }

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(naive_query_key, cl::NullRange,
                cl::NDRange{native_query_key_gws}, cl::NDRange{wgSize});
            queue.enqueueNDRangeKernel(naive_softmax, cl::NullRange,
                cl::NDRange{naive_softmax_gws}, cl::NDRange{wgSize});
            queue.enqueueNDRangeKernel(naive_value, cl::NullRange,
                cl::NDRange{naive_value_gws}, cl::NDRange{wgSize});
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> sw_time = end - start;
            auto elapsed = sw_time.count();
            best = std::min(best, elapsed);
        }
        printf("Best in %f seconds\n", best);

        if (validate) {
            printf("Checking results: preatt... "); fflush(stdout);
            std::vector<float> preatt_check(preatt_ref.size());
            queue.enqueueReadBuffer(preatt, CL_TRUE, 0, preatt_check.size() * sizeof(preatt_check[0]), preatt_check.data());
            check_results(preatt_check, preatt_ref);

            printf("Checking results: att... "); fflush(stdout);
            std::vector<float> att_check(att_ref.size());
            queue.enqueueReadBuffer(att, CL_TRUE, 0, att_check.size() * sizeof(att_check[0]), att_check.data());
            check_results(preatt_check, preatt_ref);

            printf("Checking results: out... "); fflush(stdout);
            std::vector<float> out_check(out_ref.size());
            queue.enqueueReadBuffer(out, CL_TRUE, 0, out_check.size() * sizeof(out_check[0]), out_check.data());
            check_results(out_check, out_ref);

            printf(" done!\n");
        }
    }
}

// This is a port of the minimal flash attention, see:
// https://github.com/tspeterkim/flash-attention-minimal
static void flash_attention_forward(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& out,
    cl::Buffer& inp,
    size_t wgSize,
    const std::vector<float>& out_ref)
{
    printf("%80s: ", __FUNCTION__); fflush(stdout);

    cl::Kernel flash_attention{program, "flash_attention"};
    cl::Kernel permute{program, "permute"};
    cl::Kernel unpermute{program, "unpermute"};
    if (flash_attention() == nullptr ||
        permute() == nullptr ||
        unpermute() == nullptr) {
        printf("unsupported.\n");
    } else {
        // Preprocessing: permute the single input QKV tensor into separate Q, K, and V tensors.
        // This could alternatively be done on the host.
        cl::Buffer Q{context, CL_MEM_READ_WRITE, B * T * C * sizeof(float)};
        cl::Buffer K{context, CL_MEM_READ_WRITE, B * T * C * sizeof(float)};
        cl::Buffer V{context, CL_MEM_READ_WRITE, B * T * C * sizeof(float)};

        size_t permute_gws = B * T * C;
        permute.setArg(0, Q);
        permute.setArg(1, K);
        permute.setArg(2, V);
        permute.setArg(3, inp);
        queue.enqueueNDRangeKernel(permute, cl::NullRange, cl::NDRange{permute_gws});

        // Start of the port is here:
        const int Bc = 32; const int Br = 32;
        const int nh = NH;
        const int N = T;
        const int d = C / NH;

        const int Tc = std::ceil((float) N / Bc); const int Tr = std::ceil((float) N / Br);
        const float softmax_scale = 1.0 / std::sqrt(d);

        cl::Buffer l{context, CL_MEM_READ_WRITE, B * nh * N * sizeof(float)};
        cl::Buffer m{context, CL_MEM_READ_WRITE, B * nh * N * sizeof(float)};
        queue.enqueueFillBuffer(l,         0, 0, B * nh * N * sizeof(float));
        queue.enqueueFillBuffer(m, -10000.0f, 0, B * nh * N * sizeof(float));

        cl::Buffer O{context, CL_MEM_READ_WRITE, B * T * C * sizeof(float)};
        queue.enqueueFillBuffer(O, 0, 0, B * T * C * sizeof(float));

        const int requiredLocalMemSize =
            (Br * d +   // Qi
             Bc * d +   // Kj
             Bc * d +   // Vj
             Bc * Br    // S
            ) * sizeof(float);
        cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
        cl_ulong maxLocalMemSize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
        if (requiredLocalMemSize > maxLocalMemSize) {
            printf("Device max local memory size: %" PRIu64 ", required local memory size: %d\n",
                maxLocalMemSize, requiredLocalMemSize);
        }

        // TODO: Should this be Bc or Br?
        // Doesn't matter in practice right now, but could in the future.
        cl::NDRange flash_attention_gws(B * Bc, nh);  // batch_size x num_heads
        cl::NDRange flash_attention_lws(Bc);  // Bc threads per block
        flash_attention.setArg(0, Q);
        flash_attention.setArg(1, K);
        flash_attention.setArg(2, V);
        flash_attention.setArg(3, N);
        flash_attention.setArg(4, d);
        flash_attention.setArg(5, Tc);
        flash_attention.setArg(6, Tr);
        flash_attention.setArg(7, Bc);
        flash_attention.setArg(8, Br);
        flash_attention.setArg(9, softmax_scale);
        flash_attention.setArg(10, l);
        flash_attention.setArg(11, m);
        flash_attention.setArg(12, O);
        flash_attention.setArg(13, cl::Local(requiredLocalMemSize));

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(flash_attention, cl::NullRange,
                cl::NDRange{flash_attention_gws}, cl::NDRange{flash_attention_lws});
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> sw_time = end - start;
            auto elapsed = sw_time.count();
            best = std::min(best, elapsed);
        }
        printf("Best in %f seconds\n", best);

        if (!skipinit) {
            queue.enqueueFillBuffer(out, 0, 0, out_ref.size() * sizeof(out_ref[0]));
        }

        // Postprocessing: unpermute the output tensor.
        size_t unpermute_gws = B * T * C;
        unpermute.setArg(0, O);
        unpermute.setArg(1, out);
        queue.enqueueNDRangeKernel(unpermute, cl::NullRange, cl::NDRange{unpermute_gws});

        if (validate) {
            printf("Checking results: out... "); fflush(stdout);
            std::vector<float> out_check(out_ref.size());
            queue.enqueueReadBuffer(out, CL_TRUE, 0, out_check.size() * sizeof(out_check[0]), out_check.data());
            check_results(out_check, out_ref);
            printf(" done!\n");
        }
    }
}

int main(int argc, char** argv)
{
    int platformIndex = 0;
    int deviceIndex = 0;

    std::string fileName("flashattention_kernels.cl");
    std::string buildOptions;
    size_t wgSize = 32;

    size_t mask = ~0;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<std::string>>("", "file", "Kernel File Name", fileName, &fileName);
        op.add<popl::Value<std::string>>("", "options", "Program Build Options", buildOptions, &buildOptions);
        op.add<popl::Value<size_t>>("w", "wgsize", "Work-Group Size", wgSize, &wgSize);
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
                "Usage: flashattention [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platformIndex >= platforms.size()) {
        printf("Requested platform index is %d, but only %zu platforms were found.\n",
            platformIndex, platforms.size());
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
    buildOptions += " -DEMULATE_tN16=" + std::to_string(emulate_tN16);
    buildOptions += " -DB=" + std::to_string(B);
    buildOptions += " -DT=" + std::to_string(T);
    buildOptions += " -DC=" + std::to_string(C);
    buildOptions += " -DNH=" + std::to_string(NH);

    printf("Config:\n");
    printf("\tTest Iterations: %d\n", testIterations);
    printf("\tWork-group Size: %zu\n", wgSize);
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

    std::vector<float> out_vec   (B * T * C);
    std::vector<float> preatt_vec(B * NH * T * T);
    std::vector<float> att_vec   (B * NH * T * T);
    std::vector<float> inp_vec   (B * T * 3 * C);

    printf("Initializing input...\n");
    fill_input(inp_vec, B * T * 3 * C);

    if (validate) {
        printf("Computing reference...\n");
        attention_forward_cpu(out_vec, preatt_vec, att_vec, inp_vec);
    }

    printf("Creating source buffers...\n");
    cl::Buffer out      {context, CL_MEM_READ_WRITE, B * T * C      * sizeof(float)};
    cl::Buffer vaccum   {context, CL_MEM_READ_WRITE, B * T * C      * sizeof(float)};
    cl::Buffer qkvr     {context, CL_MEM_READ_WRITE, B * T * 3 * C  * sizeof(float)};
    cl::Buffer preatt   {context, CL_MEM_READ_WRITE, B * NH * T * T * sizeof(float)};
    cl::Buffer att      {context, CL_MEM_READ_WRITE, B * NH * T * T * sizeof(float)};
    cl::Buffer inp      {context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                                                     B * T * 3 * C  * sizeof(float), inp_vec.data()};

    printf("Running tests...\n");

    if (mask & 0x1) {
        naive_attention_forward(context, program, queue, out, preatt, att, inp, wgSize, out_vec, preatt_vec, att_vec);
    }

    if (mask & 0x2) {
        flash_attention_forward(context, program, queue, out, inp, wgSize, out_vec);
    }

    printf("Done.\n");

    return 0;
}
