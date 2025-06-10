/*
// Copyright (c) 2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_layer.h>

#include <spirv/unified1/spirv.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <map>
#include <string>
#include <vector>

#include "layer_util.hpp"

#include "emulate.h"

static constexpr cl_version version_cl_khr_subgroup_queries =
    CL_MAKE_VERSION(0, 1, 0);

struct SDeviceInfo
{
    bool supports_cl_khr_subgroup_queries = true;

    std::vector<const char*>    ExtendedInstructionSets;
    std::vector<const char*>    Extensions;
    std::vector<cl_uint>        Capabilities;
};

struct SLayerContext
{
    SLayerContext()
    {
        cl_uint numPlatforms = 0;
        g_pNextDispatch->clGetPlatformIDs(
            0,
            nullptr,
            &numPlatforms);

        std::vector<cl_platform_id> platforms;
        platforms.resize(numPlatforms);
        g_pNextDispatch->clGetPlatformIDs(
            numPlatforms,
            platforms.data(),
            nullptr);

        for (auto platform : platforms) {
            getSPIRVQueriesForPlatform(platform);
        }
    }

    const SDeviceInfo& getDeviceInfo(cl_device_id device) const
    {
        // TODO: query the parent device if this is a sub-device?
        return m_DeviceInfo.at(device);
    }

    const SDeviceInfo& getDeviceInfo(cl_device_id device)
    {
        return m_DeviceInfo[device];
    }

private:
    std::map<cl_device_id, SDeviceInfo>     m_DeviceInfo;

    void getSPIRVQueriesForPlatform(cl_platform_id platform)
    {
        cl_uint numDevices = 0;
        g_pNextDispatch->clGetDeviceIDs(
            platform,
            CL_DEVICE_TYPE_ALL,
            0,
            nullptr,
            &numDevices);

        std::vector<cl_device_id> devices;
        devices.resize(numDevices);
        g_pNextDispatch->clGetDeviceIDs(
            platform,
            CL_DEVICE_TYPE_ALL,
            numDevices,
            devices.data(),
            nullptr);

        for (auto device : devices) {
            SDeviceInfo& deviceInfo = m_DeviceInfo[device];

            size_t size = 0;

            std::string deviceExtensions;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_EXTENSIONS,
                0,
                nullptr,
                &size);
            if (size) {
                deviceExtensions.resize(size);
                g_pNextDispatch->clGetDeviceInfo(
                    device,
                    CL_DEVICE_EXTENSIONS,
                    size,
                    &deviceExtensions[0],
                    nullptr);
                deviceExtensions.pop_back();
                deviceInfo.supports_cl_khr_subgroup_queries =
                    checkStringForExtension(
                        deviceExtensions.c_str(),
                        CL_KHR_SPIRV_QUERIES_EXTENSION_NAME);
            }

            std::string deviceILVersion;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_IL_VERSION,
                0,
                nullptr,
                &size);
            if (size) {
                deviceILVersion.resize(size);
                g_pNextDispatch->clGetDeviceInfo(
                    device,
                    CL_DEVICE_IL_VERSION,
                    size,
                    &deviceILVersion[0],
                    nullptr);
                deviceILVersion.pop_back();
            }

            if (deviceInfo.supports_cl_khr_subgroup_queries == false &&
                deviceILVersion.find("SPIR-V") != std::string::npos) {

                cl_version  deviceVersion = CL_MAKE_VERSION(0, 0, 0);
                std::string deviceVersionString;
                g_pNextDispatch->clGetDeviceInfo(
                    device,
                    CL_DEVICE_VERSION,
                    0,
                    nullptr,
                    &size);
                if (size) {
                    deviceVersionString.resize(size);
                    g_pNextDispatch->clGetDeviceInfo(
                        device,
                        CL_DEVICE_VERSION,
                        size,
                        &deviceVersionString[0],
                        nullptr);
                    deviceVersionString.pop_back();
                    deviceVersion = getOpenCLVersionFromString(
                        deviceVersionString.c_str());
                }

                std::string deviceProfile;
                g_pNextDispatch->clGetDeviceInfo(
                    device,
                    CL_DEVICE_PROFILE,
                    0,
                    nullptr,
                    &size);
                if (size) {
                    deviceProfile.resize(size);
                    g_pNextDispatch->clGetDeviceInfo(
                        device,
                        CL_DEVICE_PROFILE,
                        size,
                        &deviceProfile[0],
                        nullptr);
                    deviceProfile.pop_back();
                }

                cl_bool deviceImageSupport = CL_FALSE;
                g_pNextDispatch->clGetDeviceInfo(
                    device,
                    CL_DEVICE_IMAGE_SUPPORT,
                    sizeof(deviceImageSupport),
                    &deviceImageSupport,
                    nullptr);

                cl_uint deviceMaxReadWriteImageArgs = 0;
                g_pNextDispatch->clGetDeviceInfo(
                    device,
                    CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS,
                    sizeof(deviceMaxReadWriteImageArgs),
                    &deviceMaxReadWriteImageArgs,
                    nullptr);

                cl_uint deviceMaxNumSubGroups = 0;
                g_pNextDispatch->clGetDeviceInfo(
                    device,
                    CL_DEVICE_MAX_NUM_SUB_GROUPS,
                    sizeof(deviceMaxNumSubGroups),
                    &deviceMaxNumSubGroups,
                    nullptr);

                cl_bool deviceGenericAddressSpaceSupport = CL_FALSE;
                g_pNextDispatch->clGetDeviceInfo(
                    device,
                    CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT,
                    sizeof(deviceGenericAddressSpaceSupport),
                    &deviceGenericAddressSpaceSupport,
                    nullptr);

                cl_bool deviceWorkGroupCollectiveFunctionsSupport = CL_FALSE;
                g_pNextDispatch->clGetDeviceInfo(
                    device,
                    CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT,
                    sizeof(deviceWorkGroupCollectiveFunctionsSupport),
                    &deviceWorkGroupCollectiveFunctionsSupport,
                    nullptr);

                cl_bool devicePipeSupport = CL_FALSE;
                g_pNextDispatch->clGetDeviceInfo(
                    device,
                    CL_DEVICE_PIPE_SUPPORT,
                    sizeof(devicePipeSupport),
                    &devicePipeSupport,
                    nullptr);

                cl_device_device_enqueue_capabilities deviceDeviceEnqueueCapabilities = 0;
                g_pNextDispatch->clGetDeviceInfo(
                    device,
                    CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES,
                    sizeof(deviceDeviceEnqueueCapabilities),
                    &deviceDeviceEnqueueCapabilities,
                    nullptr);

                cl_device_integer_dot_product_capabilities_khr deviceIntegerDotProductCapabilities = 0;
                g_pNextDispatch->clGetDeviceInfo(
                    device,
                    CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR,
                    sizeof(deviceIntegerDotProductCapabilities),
                    &deviceIntegerDotProductCapabilities,
                    nullptr);

                cl_device_fp_atomic_capabilities_ext deviceFp32AtomicCapabilities = 0;
                g_pNextDispatch->clGetDeviceInfo(
                    device,
                    CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT,
                    sizeof(deviceFp32AtomicCapabilities),
                    &deviceFp32AtomicCapabilities,
                    nullptr);

                cl_device_fp_atomic_capabilities_ext deviceFp16AtomicCapabilities = 0;
                g_pNextDispatch->clGetDeviceInfo(
                    device,
                    CL_DEVICE_HALF_FP_ATOMIC_CAPABILITIES_EXT,
                    sizeof(deviceFp16AtomicCapabilities),
                    &deviceFp16AtomicCapabilities,
                    nullptr);

                cl_device_fp_atomic_capabilities_ext deviceFp64AtomicCapabilities = 0;
                g_pNextDispatch->clGetDeviceInfo(
                    device,
                    CL_DEVICE_DOUBLE_FP_ATOMIC_CAPABILITIES_EXT,
                    sizeof(deviceFp64AtomicCapabilities),
                    &deviceFp64AtomicCapabilities,
                    nullptr);


                // Required.
                deviceInfo.ExtendedInstructionSets.push_back("OpenCL.std");

                deviceInfo.Capabilities.push_back(spv::CapabilityAddresses);
                deviceInfo.Capabilities.push_back(spv::CapabilityFloat16Buffer);
                deviceInfo.Capabilities.push_back(spv::CapabilityInt16);
                deviceInfo.Capabilities.push_back(spv::CapabilityInt8);
                deviceInfo.Capabilities.push_back(spv::CapabilityKernel);
                deviceInfo.Capabilities.push_back(spv::CapabilityLinkage);
                deviceInfo.Capabilities.push_back(spv::CapabilityVector16);

                // Required for FULL_PROFILE devices, or devices supporting cles_khr_int64.
                if (deviceProfile == "FULL_PROFILE" ||
                    checkStringForExtension(deviceExtensions.c_str(), "cles_khr_int64")) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityInt64);
                }

                // Required for devices supporting images.
                if (deviceImageSupport == CL_TRUE) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityImage1D);
                    deviceInfo.Capabilities.push_back(spv::CapabilityImageBasic);
                    deviceInfo.Capabilities.push_back(spv::CapabilityImageBuffer);
                    deviceInfo.Capabilities.push_back(spv::CapabilityLiteralSampler);
                    deviceInfo.Capabilities.push_back(spv::CapabilitySampled1D);
                    deviceInfo.Capabilities.push_back(spv::CapabilitySampledBuffer);
                }

                // Required for devices supporting SPIR-V 1.6.
                if (deviceILVersion.find("SPIR-V_1.6") != std::string::npos) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityUniformDecoration);
                }

                // Required for devices supporting images, for OpenCL 2.0, OpenCL 2.1, OpenCL 2.2, or OpenCL 3.0 devices supporting read-write images.
                if (deviceImageSupport == CL_TRUE &&
                    (deviceVersion == CL_MAKE_VERSION(2, 0, 0) ||
                     deviceVersion == CL_MAKE_VERSION(2, 1, 0) ||
                     deviceVersion == CL_MAKE_VERSION(2, 2, 0) ||
                     (deviceVersion >= CL_MAKE_VERSION(3, 0, 0) &&
                         deviceMaxReadWriteImageArgs != 0))) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityImageReadWrite);
                }

                // Required for OpenCL 2.0, OpenCL 2.1, OpenCL 2.2, or OpenCL 3.0 devices supporting the generic address space.
                if (deviceVersion == CL_MAKE_VERSION(2, 0, 0) ||
                    deviceVersion == CL_MAKE_VERSION(2, 1, 0) ||
                    deviceVersion == CL_MAKE_VERSION(2, 2, 0) ||
                    (deviceVersion >= CL_MAKE_VERSION(3, 0, 0) &&
                        deviceGenericAddressSpaceSupport == CL_TRUE)) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityGenericPointer);
                }

                // Required for OpenCL 2.0, OpenCL 2.1, OpenCL 2.2, or OpenCL 3.0 devices supporting sub-groups or work-group collective functions.
                if (deviceVersion == CL_MAKE_VERSION(2, 0, 0) ||
                    deviceVersion == CL_MAKE_VERSION(2, 1, 0) ||
                    deviceVersion == CL_MAKE_VERSION(2, 2, 0) ||
                    (deviceVersion >= CL_MAKE_VERSION(3, 0, 0) &&
                        (deviceMaxNumSubGroups != 0 || deviceWorkGroupCollectiveFunctionsSupport == CL_TRUE))) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityGroups);
                }

                // Required for OpenCL 2.0, OpenCL 2.1, OpenCL 2.2, or OpenCL 3.0 devices supporting the generic address space.
                if (deviceVersion == CL_MAKE_VERSION(2, 0, 0) ||
                    deviceVersion == CL_MAKE_VERSION(2, 1, 0) ||
                    deviceVersion == CL_MAKE_VERSION(2, 2, 0) ||
                    (deviceVersion >= CL_MAKE_VERSION(3, 0, 0) &&
                        devicePipeSupport == CL_TRUE)) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityPipes);
                }

                // Required for OpenCL 2.0, OpenCL 2.1, OpenCL 2.2, or OpenCL 3.0 devices supporting device-side enqueue.
                if (deviceVersion == CL_MAKE_VERSION(2, 0, 0) ||
                    deviceVersion == CL_MAKE_VERSION(2, 1, 0) ||
                    deviceVersion == CL_MAKE_VERSION(2, 2, 0) ||
                    (deviceVersion >= CL_MAKE_VERSION(3, 0, 0) &&
                        deviceDeviceEnqueueCapabilities != 0)) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityDeviceEnqueue);
                }

                // Required for OpenCL 2.2 devices.
                if (deviceVersion == CL_MAKE_VERSION(2, 2, 0)) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityPipeStorage);
                }

                // Required for OpenCL 2.2, or OpenCL 3.0 devices supporting sub-groups.
                if (deviceVersion == CL_MAKE_VERSION(2, 2, 0) ||
                    (deviceVersion >= CL_MAKE_VERSION(3, 0, 0) && deviceMaxNumSubGroups != 0)) {
                    deviceInfo.Capabilities.push_back(spv::CapabilitySubgroupDispatch);
                }

                // Required for devices supporting cl_khr_expect_assume.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_expect_assume")) {
                    deviceInfo.Extensions.push_back("SPV_KHR_expect_assume");
                    deviceInfo.Capabilities.push_back(spv::CapabilityExpectAssumeKHR);
                }

                // Required for devices supporting cl_khr_extended_bit_ops.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_extended_bit_ops")) {
                    deviceInfo.Extensions.push_back("SPV_KHR_bit_instructions");
                    deviceInfo.Capabilities.push_back(spv::CapabilityBitInstructions);
                }

                // Required for devices supporting half-precision floating-point (cl_khr_fp16).
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_fp16")) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityFloat16);
                }

                // Required for devices supporting double-precision floating-point (cl_khr_fp64).
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_fp64")) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityFloat64);
                }

                // Required for devices supporting 64-bit atomics (cl_khr_int64_base_atomics or cl_khr_int64_extended_atomics).
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_int64_base_atomics") ||
                    checkStringForExtension(deviceExtensions.c_str(), "cl_khr_int64_extended_atomics")) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityInt64Atomics);
                }

                // Required for devices supporting cl_khr_integer_dot_product.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_integer_dot_product")) {
                    deviceInfo.Extensions.push_back("SPV_KHR_integer_dot_product");
                    deviceInfo.Capabilities.push_back(spv::CapabilityDotProduct);
                    deviceInfo.Capabilities.push_back(spv::CapabilityDotProductInput4x8BitPacked);
                }

                // Required for devices supporting cl_khr_integer_dot_product and CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_integer_dot_product") &&
                    (deviceIntegerDotProductCapabilities & CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR)) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityDotProductInput4x8Bit);
                }

                // Required for devices supporting cl_khr_kernel_clock.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_kernel_clock")) {
                    deviceInfo.Extensions.push_back("SPV_KHR_shader_clock");
                    deviceInfo.Capabilities.push_back(spv::CapabilityShaderClockKHR);
                }

                // Required for devices supporting both cl_khr_mipmap_image and cl_khr_mipmap_image_writes.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_mipmap_image") &&
                    checkStringForExtension(deviceExtensions.c_str(), "cl_khr_mipmap_image_writes")) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityImageMipmap);
                }

                // Required for devices supporting cl_khr_spirv_extended_debug_info.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_spirv_extended_debug_info")) {
                    deviceInfo.ExtendedInstructionSets.push_back("OpenCL.DebugInfo.100");
                }

                // Required for devices supporting cl_khr_spirv_linkonce_odr.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_spirv_linkonce_odr")) {
                    deviceInfo.Extensions.push_back("SPV_KHR_linkonce_odr");
                }

                // Required for devices supporting cl_khr_spirv_no_integer_wrap_decoration.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_spirv_no_integer_wrap_decoration")) {
                    deviceInfo.Extensions.push_back("SPV_KHR_no_integer_wrap_decoration");
                }

                // Required for devices supporting cl_khr_subgroup_ballot.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_subgroup_ballot")) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityGroupNonUniformBallot);
                }

                // Required for devices supporting cl_khr_subgroup_clustered_reduce.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_subgroup_clustered_reduce")) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityGroupNonUniformClustered);
                }

                // Required for devices supporting cl_khr_subgroup_named_barrier.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_subgroup_named_barrier")) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityNamedBarrier);
                }

                // Required for devices supporting cl_khr_subgroup_non_uniform_arithmetic.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_subgroup_non_uniform_arithmetic")) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityGroupNonUniformArithmetic);
                }

                // Required for devices supporting cl_khr_subgroup_non_uniform_vote.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_subgroup_non_uniform_vote")) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityGroupNonUniform);
                    deviceInfo.Capabilities.push_back(spv::CapabilityGroupNonUniformVote);
                }

                // Required for devices supporting cl_khr_subgroup_rotate.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_subgroup_rotate")) {
                    deviceInfo.Extensions.push_back("SPV_KHR_subgroup_rotate");
                    deviceInfo.Capabilities.push_back(spv::CapabilityGroupNonUniformRotateKHR);
                }

                // Required for devices supporting cl_khr_subgroup_shuffle.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_subgroup_shuffle")) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityGroupNonUniformShuffle);
                }

                // Required for devices supporting cl_khr_subgroup_shuffle_relative.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_subgroup_shuffle_relative")) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityGroupNonUniformShuffleRelative);
                }

                // Required for devices supporting cl_khr_work_group_uniform_arithmetic.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_khr_work_group_uniform_arithmetic")) {
                    deviceInfo.Extensions.push_back("SPV_KHR_uniform_group_instructions");
                    deviceInfo.Capabilities.push_back(spv::CapabilityGroupUniformArithmeticKHR);
                }

                // Required for devices supporting cl_ext_float_atomics and fp32 atomic adds.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_ext_float_atomics") &&
                    (deviceFp32AtomicCapabilities & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT | CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT))) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityAtomicFloat32AddEXT);
                }

                // Required for devices supporting cl_ext_float_atomics and fp32 atomic min and max.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_ext_float_atomics") &&
                    (deviceFp32AtomicCapabilities & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT | CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT))) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityAtomicFloat32MinMaxEXT);
                }

                // Required for devices supporting cl_ext_float_atomics and fp16 atomic adds.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_ext_float_atomics") &&
                    (deviceFp16AtomicCapabilities & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT | CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT))) {
                    deviceInfo.Extensions.push_back("SPV_EXT_shader_atomic_float16_add");
                    deviceInfo.Capabilities.push_back(spv::CapabilityAtomicFloat16AddEXT);
                }

                // Required for devices supporting cl_ext_float_atomics and fp16 atomic min and max.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_ext_float_atomics") &&
                    (deviceFp16AtomicCapabilities & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT | CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT))) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityAtomicFloat16MinMaxEXT);
                }

                // Required for devices supporting cl_ext_float_atomics and fp64 atomic adds.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_ext_float_atomics") &&
                    (deviceFp64AtomicCapabilities & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT | CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT))) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityAtomicFloat64MinMaxEXT);
                }

                // Required for devices supporting cl_ext_float_atomics and fp64 atomic min and max.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_ext_float_atomics") &&
                    (deviceFp64AtomicCapabilities & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT | CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT))) {
                    deviceInfo.Capabilities.push_back(spv::CapabilityAtomicFloat64MinMaxEXT);
                }

                // Required for devices supporting cl_ext_float_atomics and fp16, fp32, or fp64 atomic min or max.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_ext_float_atomics") &&
                    ((deviceFp32AtomicCapabilities & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT | CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)) ||
                     (deviceFp16AtomicCapabilities & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT | CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)) ||
                     (deviceFp64AtomicCapabilities & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT | CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)))) {
                    deviceInfo.Extensions.push_back("SPV_EXT_shader_atomic_float_min_max");
                }

                // Required for devices supporting cl_ext_float_atomics and fp32 or fp64 atomic adds.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_ext_float_atomics") &&
                    ((deviceFp32AtomicCapabilities & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT | CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)) ||
                     (deviceFp64AtomicCapabilities & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT | CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)))) {
                    deviceInfo.Extensions.push_back("SPV_EXT_shader_atomic_float_add");
                }

                // Required for devices supporting cl_intel_bfloat16_conversions.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_intel_bfloat16_conversions")) {
                    deviceInfo.Extensions.push_back("SPV_INTEL_bfloat16_conversion");
                    deviceInfo.Capabilities.push_back(spv::CapabilityBFloat16ConversionINTEL);
                }

                // Required for devices supporting cl_intel_spirv_device_side_avc_motion_estimation.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_intel_spirv_device_side_avc_motion_estimation")) {
                    deviceInfo.Extensions.push_back("SPV_INTEL_device_side_avc_motion_estimation");
                    deviceInfo.Capabilities.push_back(spv::CapabilitySubgroupAvcMotionEstimationChromaINTEL);
                    deviceInfo.Capabilities.push_back(spv::CapabilitySubgroupAvcMotionEstimationINTEL);
                    deviceInfo.Capabilities.push_back(spv::CapabilitySubgroupAvcMotionEstimationIntraINTEL);
                }

                // Required for devices supporting cl_intel_spirv_media_block_io.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_intel_spirv_media_block_io")) {
                    deviceInfo.Extensions.push_back("SPV_INTEL_media_block_io");
                    deviceInfo.Capabilities.push_back(spv::CapabilitySubgroupImageMediaBlockIOINTEL);
                }

                // Required for devices supporting cl_intel_spirv_subgroups.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_intel_spirv_subgroups")) {
                    deviceInfo.Extensions.push_back("SPV_INTEL_subgroups");
                    deviceInfo.Capabilities.push_back(spv::CapabilitySubgroupBufferBlockIOINTEL);
                    deviceInfo.Capabilities.push_back(spv::CapabilitySubgroupImageBlockIOINTEL);
                    deviceInfo.Capabilities.push_back(spv::CapabilitySubgroupShuffleINTEL);
                }

                // Required for devices supporting cl_intel_split_work_group_barrier.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_intel_split_work_group_barrier")) {
                    deviceInfo.Extensions.push_back("SPV_INTEL_split_barrier");
                    deviceInfo.Capabilities.push_back(spv::CapabilitySplitBarrierINTEL);
                }

                // Required for devices supporting cl_intel_subgroup_buffer_prefetch.
                if (checkStringForExtension(deviceExtensions.c_str(), "cl_intel_subgroup_buffer_prefetch")) {
                    deviceInfo.Extensions.push_back("SPV_INTEL_subgroup_buffer_prefetch");
                    deviceInfo.Capabilities.push_back(spv::CapabilitySubgroupBufferPrefetchINTEL);
                }
            }
        }
    }
};

SLayerContext& getLayerContext(void)
{
    static SLayerContext c;
    return c;
}

bool clGetDeviceInfo_override(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret)
{
    const auto& deviceInfo = getLayerContext().getDeviceInfo(device);
    if (deviceInfo.supports_cl_khr_subgroup_queries) {
        return false;
    }

    switch(param_name) {
    case CL_DEVICE_EXTENSIONS:
        {
            size_t  size = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_EXTENSIONS,
                0,
                nullptr,
                &size );

            std::vector<char> deviceExtensions(size);
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_EXTENSIONS,
                size,
                deviceExtensions.data(),
                nullptr );

            if( checkStringForExtension(
                    deviceExtensions.data(),
                    CL_KHR_SPIRV_QUERIES_EXTENSION_NAME ) == false )
            {
                std::string newExtensions;
                newExtensions += CL_KHR_SPIRV_QUERIES_EXTENSION_NAME;

                std::string oldExtensions(deviceExtensions.data());

                // If the old extension string ends with a space ensure the
                // new extension string does too.
                if( oldExtensions.back() == ' ' )
                {
                    newExtensions += ' ';
                }
                else
                {
                    oldExtensions += ' ';
                }

                oldExtensions += newExtensions;

                auto ptr = (char*)param_value;
                cl_int errorCode = writeStringToMemory(
                    param_value_size,
                    oldExtensions.c_str(),
                    param_value_size_ret,
                    ptr );

                if( errcode_ret )
                {
                    errcode_ret[0] = errorCode;
                }
                return true;
            }
        }
        break;
    case CL_DEVICE_EXTENSIONS_WITH_VERSION:
        {
            size_t  size = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_EXTENSIONS_WITH_VERSION,
                0,
                nullptr,
                &size );

            size_t  numExtensions = size / sizeof(cl_name_version);
            std::vector<cl_name_version>    extensions(numExtensions);
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_EXTENSIONS_WITH_VERSION,
                size,
                extensions.data(),
                nullptr );

            bool found = false;
            for( const auto& extension : extensions )
            {
                if( strcmp(extension.name, CL_KHR_SPIRV_QUERIES_EXTENSION_NAME) == 0 )
                {
                    found = true;
                    break;
                }
            }

            if( found == false )
            {
                extensions.emplace_back();
                cl_name_version& extension = extensions.back();

                memset(extension.name, 0, CL_NAME_VERSION_MAX_NAME_SIZE);
                strcpy(extension.name, CL_KHR_SPIRV_QUERIES_EXTENSION_NAME);

                extension.version = version_cl_khr_subgroup_queries;

                auto ptr = (cl_name_version*)param_value;
                cl_int errorCode = writeVectorToMemory(
                    param_value_size,
                    extensions,
                    param_value_size_ret,
                    ptr );

                if( errcode_ret )
                {
                    errcode_ret[0] = errorCode;
                }
                return true;
            }
        }
        break;
    case CL_DEVICE_SPIRV_EXTENDED_INSTRUCTION_SETS_KHR:
        {
            auto ptr = (const char**)param_value;
            cl_int errorCode = writeVectorToMemory(
                param_value_size,
                deviceInfo.ExtendedInstructionSets,
                param_value_size_ret,
                ptr);
            if (errcode_ret) {
                errcode_ret[0] = errorCode;
            }
            return true;
        }
    case CL_DEVICE_SPIRV_EXTENSIONS_KHR:
        {
            auto ptr = (const char**)param_value;
            cl_int errorCode = writeVectorToMemory(
                param_value_size,
                deviceInfo.Extensions,
                param_value_size_ret,
                ptr);
            if (errcode_ret) {
                errcode_ret[0] = errorCode;
            }
            return true;
        }
    case CL_DEVICE_SPIRV_CAPABILITIES_KHR:
        {
            auto ptr = (cl_uint*)param_value;
            cl_int errorCode = writeVectorToMemory(
                param_value_size,
                deviceInfo.Capabilities,
                param_value_size_ret,
                ptr);
            if (errcode_ret) {
                errcode_ret[0] = errorCode;
            }
            return true;
        }
    default: break;
    }

    return false;
}

bool clGetPlatformInfo_override(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret)
{
    switch(param_name) {
    case CL_PLATFORM_EXTENSIONS:
        {
            size_t  size = 0;
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_EXTENSIONS,
                0,
                nullptr,
                &size );

            std::vector<char> platformExtensions(size);
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_EXTENSIONS,
                size,
                platformExtensions.data(),
                nullptr );

            if( checkStringForExtension(
                    platformExtensions.data(),
                CL_KHR_SPIRV_QUERIES_EXTENSION_NAME ) == false )
            {
                std::string newExtensions;
                newExtensions += CL_KHR_SPIRV_QUERIES_EXTENSION_NAME;

                std::string oldExtensions(platformExtensions.data());

                // If the old extension string ends with a space ensure the
                // new extension string does too.
                if( oldExtensions.back() == ' ' )
                {
                    newExtensions += ' ';
                }
                else
                {
                    oldExtensions += ' ';
                }

                oldExtensions += newExtensions;

                auto ptr = (char*)param_value;
                cl_int errorCode = writeStringToMemory(
                    param_value_size,
                    oldExtensions.c_str(),
                    param_value_size_ret,
                    ptr );

                if( errcode_ret )
                {
                    errcode_ret[0] = errorCode;
                }
                return true;
            }
        }
        break;
    case CL_PLATFORM_EXTENSIONS_WITH_VERSION:
        {
            size_t  size = 0;
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_EXTENSIONS_WITH_VERSION,
                0,
                nullptr,
                &size );

            size_t  numExtensions = size / sizeof(cl_name_version);
            std::vector<cl_name_version>    extensions(numExtensions);
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_EXTENSIONS_WITH_VERSION,
                size,
                extensions.data(),
                nullptr );

            bool found = false;
            for( const auto& extension : extensions )
            {
                if( strcmp(extension.name, CL_KHR_SPIRV_QUERIES_EXTENSION_NAME) == 0 )
                {
                    found = true;
                    break;
                }
            }

            if( found == false )
            {
                extensions.emplace_back();
                cl_name_version& extension = extensions.back();

                memset(extension.name, 0, CL_NAME_VERSION_MAX_NAME_SIZE);
                strcpy(extension.name, CL_KHR_SPIRV_QUERIES_EXTENSION_NAME);

                extension.version = version_cl_khr_subgroup_queries;

                auto ptr = (cl_name_version*)param_value;
                cl_int errorCode = writeVectorToMemory(
                    param_value_size,
                    extensions,
                    param_value_size_ret,
                    ptr );

                if( errcode_ret )
                {
                    errcode_ret[0] = errorCode;
                }
                return true;
            }
        }
        break;
    default: break;
    }
    return false;
}
