/*
// Copyright (c) 2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp>
#include <spirv-tools/libspirv.hpp>

#include "util.hpp"

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

constexpr const char* cProlog = R"SPV(
               OpCapability Addresses
               OpCapability Linkage
               OpCapability Kernel
)SPV";

constexpr const char* cEpilog = R"SPV(
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %kernel "empty"
       %void = OpTypeVoid
 %kernel_sig = OpTypeFunction %void
     %kernel = OpFunction %void None %kernel_sig
      %entry = OpLabel
               OpReturn
               OpFunctionEnd
)SPV";

int main(int argc, char** argv)
{
    int platformIndex = 0;
    int deviceIndex = 0;

    bool build = false;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Switch>("b", "build", "Build the programs vs. dumping files", &build);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: spirvprobe [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (!checkPlatformIndex(platforms, platformIndex)) {
        return -1;
    }

    cl::Platform& platform = platforms[platformIndex];
    printf("Running on platform: %s\n",
        platform.getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    cl::Device& device = devices[deviceIndex];
    printf("Running on device: %s\n",
        device.getInfo<CL_DEVICE_NAME>().c_str() );

    cl::Context context{device};

    cl_version spvVersion = 0;
    auto ilVersions = device.getInfo<CL_DEVICE_ILS_WITH_VERSION>();
    for (const auto& ilVersion : ilVersions) {
        if (std::string(ilVersion.name) == std::string("SPIR-V") &&
            ilVersion.version > spvVersion) {
            spvVersion = ilVersion.version;
        }
    }

    if (spvVersion == 0) {
        printf("No supported SPIR-V versions were found, exiting.\n");
        return -1;
    }

    const int spvVersionMajor = CL_VERSION_MAJOR(spvVersion);
    const int spvVersionMinor = CL_VERSION_MINOR(spvVersion);
    printf("Highest supported SPIR-V version is: %d.%d\n",
        spvVersionMajor,
        spvVersionMinor);

    const cl_uint addressBits = device.getInfo<CL_DEVICE_ADDRESS_BITS>();
    if (addressBits != 64) {
        printf("This test requires 64-bit addresses, but CL_DEVICE_ADDRESS_BITS returned %u.\n",
            addressBits);
        return -1;
    }

    spv_target_env env = SPV_ENV_UNIVERSAL_1_0;
    if (spvVersionMajor > 1 || spvVersionMinor >= 6) {
        env = SPV_ENV_UNIVERSAL_1_6;
    } else if (spvVersionMinor == 5) {
        env = SPV_ENV_UNIVERSAL_1_5;
    } else if (spvVersionMinor == 4) {
        env = SPV_ENV_UNIVERSAL_1_4;
    } else if (spvVersionMinor == 3) {
        env = SPV_ENV_UNIVERSAL_1_3;
    } else if (spvVersionMinor == 2) {
        env = SPV_ENV_UNIVERSAL_1_2;
    } else if (spvVersionMinor == 1) {
        env = SPV_ENV_UNIVERSAL_1_1;
    }

    printf("Collecting test probes...\n");

    typedef std::map<std::string, std::vector<spv::Capability>> CProbeMap;
    CProbeMap   probes;
#define SPIRV_PROBE_EXTENSION(_e) \
    if (probes.count(#_e) == 0) { probes[#_e] = {}; }
#define SPIRV_PROBE_EXTENSION_CAPABILITY(_e, _c) \
    probes[#_e].push_back(spv::Capability##_c);
#include "spirv_probes_generated.def"
#if 0
probes["SPV_INTEL_16bit_atomics"].push_back(6260); // AtomicInt16CompareExchangeINTEL
probes["SPV_INTEL_16bit_atomics"].push_back(6261); // Int16AtomicsINTEL
probes["SPV_INTEL_16bit_atomics"].push_back(6262); // AtomicBFloat16LoadStoreINTEL
probes["SPV_INTEL_16bit_atomics"].push_back(6255); // AtomicBFloat16AddINTEL
probes["SPV_INTEL_16bit_atomics"].push_back(6256); // AtomicBFloat16MinMaxINTEL
probes["SPV_INTEL_bfloat16_arithmetic"].push_back(6226); // BFloat16ArithmeticINTEL
probes["SPV_INTEL_global_variable_decorations"].push_back(6146); // GlobalVariableDecorationsINTEL
probes["SPV_INTEL_joint_matrix"].push_back(6434); // PackedCooperativeMatrixINTEL
probes["SPV_INTEL_joint_matrix"].push_back(6435); // CooperativeMatrixInvocationInstructionsINTEL
probes["SPV_INTEL_joint_matrix"].push_back(6436); // CooperativeMatrixTF32ComponentTypeINTEL
probes["SPV_INTEL_joint_matrix"].push_back(6437); // CooperativeMatrixBFloat16ComponentTypeINTEL
probes["SPV_INTEL_joint_matrix"].push_back(6411); // CooperativeMatrixPrefetchINTEL
probes["SPV_INTEL_sigmoid"].push_back(6167); // SigmoidINTEL
// TODO SPV_INTEL_​subgroup_​matrix_​multiply_​accumulate_​float4
// TODO SPV_INTEL_​subgroup_​matrix_​multiply_​accumulate_​float8
probes["SPV_INTEL_subgroup_scaled_matrix_multiply_accumulate"].push_back(6263); // SubgroupScaledMatrixMultiplyAccumulateINTEL
#endif
#undef SPIRV_PROBE_EXTENSION
#undef SPIRV_PROBE_EXTENSION_CAPABILITY

    std::set<std::string> skipExtensions;
    skipExtensions.insert("SPV_ARM_graph");                 // No OpGraphEntryPointARM instruction was found but the GraphARM capability is declared.
    skipExtensions.insert("SPV_KHR_vulkan_memory_model");   // VulkanMemoryModelKHR capability must only be specified if the VulkanKHR memory model is used.
    skipExtensions.insert("SPV_NV_bindless_texture");       // Missing required OpSamplerImageAddressingModeNV instruction.

    printf("Starting testing!\n\n");

    auto DisMessagePrinter =
        [](spv_message_level_t, const char *, const spv_position_t &,
           const char *message) -> void { fprintf(stderr, "spirv error: %s\n", message); };

    CProbeMap::iterator i = probes.begin();
    while (i != probes.end()) {
        const auto& extension = (*i).first;
        const auto& capabilities = (*i).second;

        ++i;

        if (skipExtensions.count(extension)) {
            printf("Skipped extension %s.\n", extension.c_str());
            continue;
        }

        for (auto capability : capabilities) {
            printf("Testing extension %s with capability %s (%d)... ",
                extension.c_str(),
                spv::CapabilityToString(capability),
                static_cast<int>(capability));
            fflush(stdout);

            std::vector<uint32_t> spirvBinary;

            spvtools::SpirvTools tools(env);
            tools.SetMessageConsumer(DisMessagePrinter);

            std::string spirv_text;
            spirv_text += cProlog;
            spirv_text += "               OpCapability ";
            spirv_text += spv::CapabilityToString(capability);
            spirv_text += "\n";
            spirv_text += "               OpExtension \"";
            spirv_text += extension;
            spirv_text += "\"\n";
            spirv_text += cEpilog;

            if (!tools.Assemble(spirv_text, &spirvBinary))
            {
                printf("failed to assemble!\n");
                continue;
            }

            if (!tools.Validate(spirvBinary.data(), spirvBinary.size())) {
                printf("failed to validate!\n");
                continue;
            }

            if (build) {
                cl::Program program{
                    clCreateProgramWithIL(
                        context(),
                        spirvBinary.data(),
                        spirvBinary.size() * sizeof(uint32_t),
                        nullptr)};
                cl_int errorCode = program.build();
                if (errorCode != CL_SUCCESS) {
                    printf("failed to build!\n");
                    printf("%s\n", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str());
                    continue;
                }
            } else {
                std::string filename("./spirv_dumps/");
                filename += extension;
                filename += "_";
                filename += spv::CapabilityToString(capability);
                filename += ".spv";

                std::ofstream os;
                os.open(filename, std::ios::out | std::ios::binary);
                if (os.good()) {
                    os.write(
                        (const char*)spirvBinary.data(),
                        spirvBinary.size() * sizeof(uint32_t));
                    os.close();
                } else {
                    printf("Failed to dump SPIR-V to file %s!\n", filename.c_str());
                }
            }

            printf("success.\n");
        }
    }

    return 0;
}