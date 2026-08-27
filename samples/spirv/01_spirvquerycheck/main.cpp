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

bool do_test(
    const cl::Context& context,
    spv_target_env env,
    const std::string& extension,
    const std::string& instruction_set,
    spv::Capability capability)
{
    auto DisMessagePrinter =
        [](spv_message_level_t, const char *, const spv_position_t &,
           const char *message) -> void { fprintf(stderr, "spirv error: %s\n", message); };

    std::vector<uint32_t> spirvBinary;

    spvtools::SpirvTools tools(env);
    tools.SetMessageConsumer(DisMessagePrinter);

    std::string spirv_text;
    spirv_text += cProlog;
    if (capability != spv::CapabilityMax) {
        spirv_text += "               OpCapability ";
        spirv_text += spv::CapabilityToString(capability);
        spirv_text += "\n";
    }
    if (!extension.empty()) {
        spirv_text += "               OpExtension \"";
        spirv_text += extension;
        spirv_text += "\"\n";
    }
    if (!instruction_set.empty()) {
        spirv_text += "        %eis = OpExtInstImport \"";
        spirv_text += instruction_set;
        spirv_text += "\"\n";
    }
    spirv_text += cEpilog;

    if (!tools.Assemble(spirv_text, &spirvBinary)) {
        printf("SPIR-V module failed to assemble! %s\n", spirv_text.c_str());
        return false;
    }

    if (!tools.Validate(spirvBinary.data(), spirvBinary.size())) {
        printf("SPIR-V module failed to validate! %s\n", spirv_text.c_str());
        return false;
    }

    cl::Program program{
        clCreateProgramWithIL(
            context(),
            spirvBinary.data(),
            spirvBinary.size() * sizeof(uint32_t),
            nullptr)};
    cl_int errorCode = program.build();
    if (errorCode != CL_SUCCESS) {
        printf("SPIR-V program failed to build!\n");
        for (const auto& device : context.getInfo<CL_CONTEXT_DEVICES>()) {
            printf("--- Build log for device %s:\n", device.getInfo<CL_DEVICE_NAME>().c_str());
            printf("%s\n\n", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str());
        }
        return false;
    }

    return true;
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
                "Usage: spirvquerycheck [options]\n"
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
    } else if (spvVersionMajor == 1 && spvVersionMinor == 5) {
        env = SPV_ENV_UNIVERSAL_1_5;
    } else if (spvVersionMajor == 1 && spvVersionMinor == 4) {
        env = SPV_ENV_UNIVERSAL_1_4;
    } else if (spvVersionMajor == 1 && spvVersionMinor == 3) {
        env = SPV_ENV_UNIVERSAL_1_3;
    } else if (spvVersionMajor == 1 && spvVersionMinor == 2) {
        env = SPV_ENV_UNIVERSAL_1_2;
    } else if (spvVersionMajor == 1 && spvVersionMinor == 1) {
        env = SPV_ENV_UNIVERSAL_1_1;
    } else {
        env = SPV_ENV_UNIVERSAL_1_0;
    }

    size_t failures = 0;
    size_t warnings = 0;

    printf("\n\nChecking queried extended instruction sets...\n");
    auto spirvExtendedInstructionSets =
        device.getInfo<CL_DEVICE_SPIRV_EXTENDED_INSTRUCTION_SETS_KHR>();
    for (auto s : spirvExtendedInstructionSets) {
        printf("Checking extended instruction set: %s\n", s);
        failures += do_test(context, env, "", s, spv::CapabilityMax) ? 0 : 1;
    }

    printf("\n\nChecking queried extensions...\n");
    auto spirvExtensions =
        device.getInfo<CL_DEVICE_SPIRV_EXTENSIONS_KHR>();
    for (auto s : spirvExtensions) {
        printf("Checking extension: %s\n", s);
        failures += do_test(context, env, s, "", spv::CapabilityMax) ? 0 : 1;
    }

    printf("\n\nCollecting SPIR-V capability dependencies...\n");
    struct CapabilityDependencies
    {
        std::vector<std::string> extensions;
        std::string version;
    };

    std::map<spv::Capability, CapabilityDependencies> dependencies;

#define SPIRV_CAPABILITY_VERSION_DEPENDENCY(_cap, _ver)                        \
    dependencies[spv::Capability##_cap].version = _ver;
#define SPIRV_CAPABILITY_EXTENSION_DEPENDENCY(_cap, _ext)                      \
    dependencies[spv::Capability##_cap].extensions.push_back(_ext);
#include "spirv_capability_deps.def"

    const std::string spvVersions = device.getInfo<CL_DEVICE_IL_VERSION>();

    printf("\n\nChecking queried capabilities...\n");
    auto spirvCapabilities =
        device.getInfo<CL_DEVICE_SPIRV_CAPABILITIES_KHR>();
    for (auto c : spirvCapabilities) {
        printf("Checking capability: %s (%d)\n",
            spv::CapabilityToString(static_cast<spv::Capability>(c)),
            static_cast<int>(c));

        auto it = dependencies.find(static_cast<spv::Capability>(c));
        if (it != dependencies.end()) {
            bool found = false;
            if (!it->second.version.empty()) {
                if (spvVersions.find(it->second.version) != std::string::npos) {
                    printf("    with SPIR-V version: %s\n", it->second.version.c_str());
                    found = true;
                    failures += do_test(context, env, "", "", static_cast<spv::Capability>(c)) ? 0 : 1;
                }
            }
            for (const auto& ext : it->second.extensions) {
                for (const auto& supported : spirvExtensions) {
                    if (ext == supported) {
                        printf("    with SPIR-V extension: %s\n", ext.c_str());
                        found = true;
                        failures += do_test(context, env, ext, "", static_cast<spv::Capability>(c)) ? 0 : 1;
                    }
                }
            }
            if (!found) {
                failures++;
                printf("Required dependency for capability %s (%d) not found!\n",
                    spv::CapabilityToString(static_cast<spv::Capability>(c)),
                    static_cast<int>(c));
                if (!it->second.version.empty()) {
                    printf("  Looked for SPIR-V version: %s\n", it->second.version.c_str());
                }
                for (const auto& ext : it->second.extensions) {
                    printf("  Looked for extension: %s\n", ext.c_str());
                }
            }
        } else {
            warnings++;
            printf("Dependency information for capability %s (%d) not found!\n",
                spv::CapabilityToString(static_cast<spv::Capability>(c)),
                static_cast<int>(c));
        }
    }

    printf("\n\nTest complete: %zu failure(s), %zu warning(s).\n", failures, warnings);
    return 0;
}