/* Test the capabilities of the platform to use host-accessible device memory
 * allocations for communicating while the kernel is running. */

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <iostream>

//#define FINE_GRAIN_SVM
#define UNIFIED_SHARED_MEMORY

#define CPU_FENCE do {                                 \
    std::atomic_thread_fence(std::memory_order_seq_cst);  \
    __builtin_ia32_mfence();                              \
    __builtin_ia32_sfence();                              \
  } while (0)

#define CPU_CACHE_LINE_FLUSH(ADDR) __builtin_ia32_clflush((const void*)ADDR)

bool DeviceHostPingPongTest();

int main(int argc, char** argv) {
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
                "Usage: pingpong [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  printf("Running on platform: %s\n",
      platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

  cl::Platform::setDefault(platforms[platformIndex]);

  std::vector<cl::Device> devices;
  platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

  printf("Running on device: %s\n",
      devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

  cl::Device::setDefault(devices[deviceIndex]);

  if (DeviceHostPingPongTest())
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}

bool DeviceHostPingPongTest()
{
  // Raw string literal for the second kernel
  std::string pingPongKernelSrc{R"CLC(
   kernel void pingPong(global volatile atomic_int *hostAccessibleDeviceMem)
   {
     int val;
     while ((val = atomic_load(hostAccessibleDeviceMem)) > 0)
     {
       if (val % 2 == 0)
       {
         // If I the printfs are enabled, results with
         // NDRange failure -5 (CL_OUT_OF_RESOURCES) on iGPU.

         // If disabled, gets stuck; seems the data does not propagate through
         // caches etc. after the initial sych., most likely).

         //printf("Device updating %p...\n", hostAccessibleDeviceMem);
         val -= 1;
         atomic_store(hostAccessibleDeviceMem, val);
         //printf("Device value now at %d.\n", *hostAccessibleDeviceMem);
       }
     }
   }
)CLC"};

  std::vector<std::string> programStrings;
  programStrings.push_back(pingPongKernelSrc);

  cl::Program prog(programStrings);
  try {
    prog.build("-cl-std=CL3.0");
  } catch (...) {
    // Print build info for all devices
    cl_int buildErr = CL_SUCCESS;
    auto buildInfo = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
    for (auto &pair : buildInfo) {
      std::cerr << pair.second << std::endl << std::endl;
    }
    return false;
  }

  // shared data allocations

#ifdef FINE_GRAIN_SVM
  volatile int* pingPongValue = cl::allocate_svm<int, cl::SVMTraitCoarse<>>().get();

#elif defined(UNIFIED_SHARED_MEMORY)
  cl_int errcode;
  cl_mem_properties_intel properties[] =
    {CL_MEM_ALLOC_FLAGS_INTEL, CL_MEM_ALLOC_WRITE_COMBINED_INTEL, 0};
  volatile int *pingPongValue =
    (int*)clSharedMemAllocINTEL(cl::Context::getDefault().get(),
                           cl::Device::getDefault().get(),
                           properties, sizeof(int),
                           4, &errcode);
  if (pingPongValue == nullptr || errcode != CL_SUCCESS) {
    std::cerr << "USM allocation failure! Errcode " << errcode << std::endl;
    return false;
  }
#else

#error You need to specify the SVM allocation API.

#endif

  *pingPongValue = 10;
  CPU_FENCE;

  //////////////
  // Traditional cl_mem allocations

  auto pingPongKernel =
    cl::KernelFunctor<cl_int*>(prog, "pingPong");

#ifdef UNIFIED_SHARED_MEMORY
  cl_bool T = CL_TRUE;
  clSetKernelExecInfo(pingPongKernel.getKernel().get(),
                      CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL,
                      sizeof(cl_bool), &T);
#endif

  pingPongKernel(cl::EnqueueArgs(cl::NDRange(1), cl::NDRange(1)), (int*)pingPongValue);

  while (*pingPongValue > 0)
  {
    if (*pingPongValue % 2 == 1)
    {
      printf("Host updating %p...\n", pingPongValue);
      *pingPongValue -= 1;
      printf("Host value now at %d.\n", *pingPongValue);
    }
    CPU_FENCE;
    CPU_CACHE_LINE_FLUSH(pingPongValue);
  }

  std::cout << "Output: " << *pingPongValue << std::endl;

  return true;
}
