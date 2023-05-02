/*
// Copyright (c) 2022 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_layer.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "layer_util.hpp"

#include "emulate.h"

static constexpr cl_version version_cl_khr_command_buffer =
    CL_MAKE_VERSION(0, 9, 2);
static constexpr cl_version version_cl_khr_command_buffer_mutable_dispatch =
    CL_MAKE_VERSION(0, 9, 0);

SLayerContext& getLayerContext(void)
{
    static SLayerContext c;
    return c;
}

#if defined(cl_khr_command_buffer_mutable_dispatch)

// Supported mutable dispatch capabilities.
// Right now, all capabilities are supported.
const cl_mutable_dispatch_fields_khr g_MutableDispatchCaps =
    CL_MUTABLE_DISPATCH_GLOBAL_OFFSET_KHR |
    CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR |
    CL_MUTABLE_DISPATCH_LOCAL_SIZE_KHR |
    CL_MUTABLE_DISPATCH_ARGUMENTS_KHR |
    CL_MUTABLE_DISPATCH_EXEC_INFO_KHR;

#endif // defined(cl_khr_command_buffer_mutable_dispatch)

typedef struct _cl_mutable_command_khr
{
    static bool isValid( cl_mutable_command_khr command )
    {
        return command && command->Magic == cMagic;
    }

    cl_command_buffer_khr   getCmdBuf() const
    {
        return CmdBuf;
    }

    cl_command_type getType() const
    {
        return Type;
    }

#if defined(cl_khr_command_buffer_mutable_dispatch)
    virtual cl_int  getInfo(
        cl_mutable_command_info_khr param_name,
        size_t param_value_size,
        void* param_value,
        size_t* param_value_size_ret)
    {
        switch( param_name )
        {
        case CL_MUTABLE_COMMAND_COMMAND_QUEUE_KHR:
            {
                auto ptr = (cl_command_queue*)param_value;
                return writeParamToMemory(
                    param_value_size,
                    Queue,
                    param_value_size_ret,
                    ptr );
            }
            break;
        case CL_MUTABLE_COMMAND_COMMAND_BUFFER_KHR:
            {
                auto ptr = (cl_command_buffer_khr*)param_value;
                return writeParamToMemory(
                    param_value_size,
                    CmdBuf,
                    param_value_size_ret,
                    ptr );
            }
            break;
        case CL_MUTABLE_COMMAND_COMMAND_TYPE_KHR:
            {
                auto ptr = (cl_command_type*)param_value;
                return writeParamToMemory(
                    param_value_size,
                    Type,
                    param_value_size_ret,
                    ptr );
            }
            break;
        // These are only valid for clCommandNDRangeKernel, but the spec says
        // they should return size = 0 rather than an error.
        case CL_MUTABLE_DISPATCH_PROPERTIES_ARRAY_KHR:
            {
                auto ptr = (cl_ndrange_kernel_command_properties_khr*)param_value;
                return writeVectorToMemory(
                    param_value_size,
                    {}, // No properties are currently supported.
                    param_value_size_ret,
                    ptr );
            }
            break;
        case CL_MUTABLE_DISPATCH_KERNEL_KHR:
            {
                auto ptr = (cl_kernel*)param_value;
                return writeVectorToMemory(
                    param_value_size,
                    {},
                    param_value_size_ret,
                    ptr );
            }
            break;
        case CL_MUTABLE_DISPATCH_DIMENSIONS_KHR:
            {
                auto ptr = (cl_uint*)param_value;
                return writeVectorToMemory(
                    param_value_size,
                    {},
                    param_value_size_ret,
                    ptr );
            }
            break;
        case CL_MUTABLE_DISPATCH_GLOBAL_WORK_OFFSET_KHR:
        case CL_MUTABLE_DISPATCH_GLOBAL_WORK_SIZE_KHR:
        case CL_MUTABLE_DISPATCH_LOCAL_WORK_SIZE_KHR:
            {
                auto ptr = (size_t*)param_value;
                return writeVectorToMemory(
                    param_value_size,
                    {},
                    param_value_size_ret,
                    ptr );
            }
            break;
        default:
            break;
        }

        return CL_INVALID_VALUE;
    }
#endif // defined(cl_khr_command_buffer_mutable_dispatch)

    void addDependencies(
        cl_uint num_sync_points,
        const cl_sync_point_khr* wait_list,
        cl_sync_point_khr sync_point)
    {
        if (SyncPoint == 0 && WaitList.empty()) {
            WaitList.assign(wait_list, wait_list + num_sync_points);
            SyncPoint = sync_point;
        }
    }

    std::vector<cl_event> getEventWaitList(
        const std::vector<cl_event>& deps) const
    {
        std::vector<cl_event> eventWaitList(WaitList.size());
        std::transform(
            WaitList.cbegin(),
            WaitList.cend(),
            eventWaitList.begin(),
            [&](cl_uint s){ return deps[s]; });
        return eventWaitList;
    }

    cl_event* getEventSignalPtr(
        std::vector<cl_event>& deps) const
    {
        return SyncPoint != 0 ? &deps[SyncPoint] : nullptr;
    }

    virtual ~_cl_mutable_command_khr() = default;

    virtual int playback(
        cl_command_queue,
        std::vector<cl_event>&) const = 0;

    _cl_mutable_command_khr(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue,
        cl_command_type type);

private:
    static constexpr cl_uint cMagic = 0x4d434442;   // "MCMD"

    const cl_uint Magic;
    const cl_command_type Type;
    cl_command_buffer_khr CmdBuf;
    cl_command_queue Queue;

    std::vector<cl_sync_point_khr> WaitList;
    cl_sync_point_khr SyncPoint = 0;
} Command;

struct BarrierWithWaitList : Command
{
    static std::unique_ptr<BarrierWithWaitList> create(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue)
    {
        auto ret = std::unique_ptr<BarrierWithWaitList>(
            new BarrierWithWaitList(cmdbuf, queue));
        return ret;
    }

    int playback(
        cl_command_queue queue,
        std::vector<cl_event>& deps) const override
    {
        auto wait_list = getEventWaitList(deps);
        auto signal = getEventSignalPtr(deps);
        return g_pNextDispatch->clEnqueueBarrierWithWaitList(
            queue,
            static_cast<cl_uint>(wait_list.size()),
            wait_list.data(),
            signal);
    }

private:
    BarrierWithWaitList(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue) : Command(cmdbuf, queue, CL_COMMAND_BARRIER) {};
};

struct CopyBuffer : Command
{
    static std::unique_ptr<CopyBuffer> create(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue,
        cl_mem src_buffer,
        cl_mem dst_buffer,
        size_t src_offset,
        size_t dst_offset,
        size_t size)
    {
        auto ret = std::unique_ptr<CopyBuffer>(
            new CopyBuffer(cmdbuf, queue));

        ret->src_buffer = src_buffer;
        ret->dst_buffer = dst_buffer;
        ret->src_offset = src_offset;
        ret->dst_offset = dst_offset;
        ret->size = size;

        g_pNextDispatch->clRetainMemObject(ret->src_buffer);
        g_pNextDispatch->clRetainMemObject(ret->dst_buffer);

        return ret;
    }

    ~CopyBuffer()
    {
        g_pNextDispatch->clReleaseMemObject(src_buffer);
        g_pNextDispatch->clReleaseMemObject(dst_buffer);
    }

    int playback(
        cl_command_queue queue,
        std::vector<cl_event>& deps) const override
    {
        auto wait_list = getEventWaitList(deps);
        auto signal = getEventSignalPtr(deps);
        return g_pNextDispatch->clEnqueueCopyBuffer(
            queue,
            src_buffer,
            dst_buffer,
            src_offset,
            dst_offset,
            size,
            static_cast<cl_uint>(wait_list.size()),
            wait_list.data(),
            signal);
    }

    cl_mem src_buffer = nullptr;
    cl_mem dst_buffer = nullptr;
    size_t src_offset = 0;
    size_t dst_offset = 0;
    size_t size = 0;

private:
    CopyBuffer(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue) : Command(cmdbuf, queue, CL_COMMAND_COPY_BUFFER) {};
};

struct CopyBufferRect : Command
{
    static std::unique_ptr<CopyBufferRect> create(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue,
        cl_mem src_buffer,
        cl_mem dst_buffer,
        const size_t* src_origin,
        const size_t* dst_origin,
        const size_t* region,
        size_t src_row_pitch,
        size_t src_slice_pitch,
        size_t dst_row_pitch,
        size_t dst_slice_pitch)
    {
        auto ret = std::unique_ptr<CopyBufferRect>(
            new CopyBufferRect(cmdbuf, queue));

        ret->src_buffer = src_buffer;
        ret->dst_buffer = dst_buffer;
        std::copy(src_origin, src_origin + 3, ret->src_origin.begin());
        std::copy(dst_origin, dst_origin + 3, ret->dst_origin.begin());
        std::copy(region, region + 3, ret->region.begin());
        ret->src_row_pitch = src_row_pitch;
        ret->src_slice_pitch = src_slice_pitch;
        ret->dst_row_pitch = dst_row_pitch;
        ret->dst_slice_pitch = dst_slice_pitch;

        g_pNextDispatch->clRetainMemObject(ret->src_buffer);
        g_pNextDispatch->clRetainMemObject(ret->dst_buffer);

        return ret;
    }

    ~CopyBufferRect()
    {
        g_pNextDispatch->clReleaseMemObject(src_buffer);
        g_pNextDispatch->clReleaseMemObject(dst_buffer);
    }

    int playback(
        cl_command_queue queue,
        std::vector<cl_event>& deps) const override
    {
        auto wait_list = getEventWaitList(deps);
        auto signal = getEventSignalPtr(deps);
        return g_pNextDispatch->clEnqueueCopyBufferRect(
            queue,
            src_buffer,
            dst_buffer,
            src_origin.data(),
            dst_origin.data(),
            region.data(),
            src_row_pitch,
            src_slice_pitch,
            dst_row_pitch,
            dst_slice_pitch,
            static_cast<cl_uint>(wait_list.size()),
            wait_list.data(),
            signal);
    }

    cl_mem src_buffer = nullptr;
    cl_mem dst_buffer = nullptr;
    std::array<size_t, 3> src_origin = {0, 0, 0};
    std::array<size_t, 3> dst_origin = {0, 0, 0};
    std::array<size_t, 3> region = {0, 0, 0};
    size_t src_row_pitch = 0;
    size_t src_slice_pitch = 0;
    size_t dst_row_pitch = 0;
    size_t dst_slice_pitch = 0;

private:
    CopyBufferRect(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue) : Command(cmdbuf, queue, CL_COMMAND_COPY_BUFFER_RECT) {};
};

struct CopyBufferToImage : Command
{
    static std::unique_ptr<CopyBufferToImage> create(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue,
        cl_mem src_buffer,
        cl_mem dst_image,
        size_t src_offset,
        const size_t* dst_origin,
        const size_t* region)
    {
        auto ret = std::unique_ptr<CopyBufferToImage>(
            new CopyBufferToImage(cmdbuf, queue));

        ret->src_buffer = src_buffer;
        ret->dst_image = dst_image;
        ret->src_offset = src_offset;
        std::copy(dst_origin, dst_origin + 3, ret->dst_origin.begin());
        std::copy(region, region + 3, ret->region.begin());

        g_pNextDispatch->clRetainMemObject(ret->src_buffer);
        g_pNextDispatch->clRetainMemObject(ret->dst_image);

        return ret;
    }

    ~CopyBufferToImage()
    {
        g_pNextDispatch->clReleaseMemObject(src_buffer);
        g_pNextDispatch->clReleaseMemObject(dst_image);
    }

    int playback(
        cl_command_queue queue,
        std::vector<cl_event>& deps) const override
    {
        auto wait_list = getEventWaitList(deps);
        auto signal = getEventSignalPtr(deps);
        return g_pNextDispatch->clEnqueueCopyBufferToImage(
            queue,
            src_buffer,
            dst_image,
            src_offset,
            dst_origin.data(),
            region.data(),
            static_cast<cl_uint>(wait_list.size()),
            wait_list.data(),
            signal);
    }

    cl_mem src_buffer = nullptr;
    cl_mem dst_image = nullptr;
    size_t src_offset = 0;
    std::array<size_t, 3> dst_origin = {0, 0, 0};
    std::array<size_t, 3> region = {0, 0, 0};

private:
    CopyBufferToImage(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue) : Command(cmdbuf, queue, CL_COMMAND_COPY_BUFFER_TO_IMAGE) {};
};

struct CopyImage : Command
{
    static std::unique_ptr<CopyImage> create(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue,
        cl_mem src_image,
        cl_mem dst_image,
        const size_t* src_origin,
        const size_t* dst_origin,
        const size_t* region)
    {
        auto ret = std::unique_ptr<CopyImage>(
            new CopyImage(cmdbuf, queue));

        ret->src_image = src_image;
        ret->dst_image = dst_image;
        std::copy(src_origin, src_origin + 3, ret->src_origin.begin());
        std::copy(dst_origin, dst_origin + 3, ret->dst_origin.begin());
        std::copy(region, region + 3, ret->region.begin());

        g_pNextDispatch->clRetainMemObject(ret->src_image);
        g_pNextDispatch->clRetainMemObject(ret->dst_image);

        return ret;
    }

    ~CopyImage()
    {
        g_pNextDispatch->clReleaseMemObject(src_image);
        g_pNextDispatch->clReleaseMemObject(dst_image);
    }

    int playback(
        cl_command_queue queue,
        std::vector<cl_event>& deps) const override
    {
        auto wait_list = getEventWaitList(deps);
        auto signal = getEventSignalPtr(deps);
        return g_pNextDispatch->clEnqueueCopyImage(
            queue,
            src_image,
            dst_image,
            src_origin.data(),
            dst_origin.data(),
            region.data(),
            static_cast<cl_uint>(wait_list.size()),
            wait_list.data(),
            signal);
    }

    cl_mem src_image = nullptr;
    cl_mem dst_image = nullptr;
    std::array<size_t, 3> src_origin = {0, 0, 0};
    std::array<size_t, 3> dst_origin = {0, 0, 0};
    std::array<size_t, 3> region = {0, 0, 0};

private:
    CopyImage(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue) : Command(cmdbuf, queue, CL_COMMAND_COPY_IMAGE) {};
};

struct CopyImageToBuffer : Command
{
    static std::unique_ptr<CopyImageToBuffer> create(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue,
        cl_mem src_image,
        cl_mem dst_buffer,
        const size_t* src_origin,
        const size_t* region,
        size_t dst_offset)
    {
        auto ret = std::unique_ptr<CopyImageToBuffer>(
            new CopyImageToBuffer(cmdbuf, queue));

        ret->src_image = src_image;
        ret->dst_buffer = dst_buffer;
        std::copy(src_origin, src_origin + 3, ret->src_origin.begin());
        std::copy(region, region + 3, ret->region.begin());
        ret->dst_offset = dst_offset;

        g_pNextDispatch->clRetainMemObject(ret->src_image);
        g_pNextDispatch->clRetainMemObject(ret->dst_buffer);

        return ret;
    }

    ~CopyImageToBuffer()
    {
        g_pNextDispatch->clReleaseMemObject(src_image);
        g_pNextDispatch->clReleaseMemObject(dst_buffer);
    }

    int playback(
        cl_command_queue queue,
        std::vector<cl_event>& deps) const override
    {
        auto wait_list = getEventWaitList(deps);
        auto signal = getEventSignalPtr(deps);
        return g_pNextDispatch->clEnqueueCopyImageToBuffer(
            queue,
            src_image,
            dst_buffer,
            src_origin.data(),
            region.data(),
            dst_offset,
            static_cast<cl_uint>(wait_list.size()),
            wait_list.data(),
            signal);
    }

    cl_mem src_image = nullptr;
    cl_mem dst_buffer = nullptr;
    std::array<size_t, 3> src_origin = {0, 0, 0};
    std::array<size_t, 3> region = {0, 0, 0};
    size_t dst_offset = 0;

private:
    CopyImageToBuffer(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue) : Command(cmdbuf, queue, CL_COMMAND_COPY_IMAGE_TO_BUFFER) {};
};

struct FillBuffer : Command
{
    static std::unique_ptr<FillBuffer> create(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue,
        cl_mem buffer,
        const void* pattern,
        size_t pattern_size,
        size_t offset,
        size_t size)
    {
        auto ret = std::unique_ptr<FillBuffer>(
            new FillBuffer(cmdbuf, queue));

        ret->buffer = buffer;

        auto p = reinterpret_cast<const uint8_t*>(pattern);
        ret->pattern.reserve(pattern_size);
        ret->pattern.insert(
            ret->pattern.begin(),
            p,
            p + pattern_size);

        ret->offset = offset;
        ret->size = size;

        g_pNextDispatch->clRetainMemObject(ret->buffer);

        return ret;
    }

    ~FillBuffer()
    {
        g_pNextDispatch->clReleaseMemObject(buffer);
    }

    int playback(
        cl_command_queue queue,
        std::vector<cl_event>& deps) const override
    {
        auto wait_list = getEventWaitList(deps);
        auto signal = getEventSignalPtr(deps);
        return g_pNextDispatch->clEnqueueFillBuffer(
            queue,
            buffer,
            pattern.data(),
            pattern.size(),
            offset,
            size,
            static_cast<cl_uint>(wait_list.size()),
            wait_list.data(),
            signal);
    }

    cl_mem buffer = nullptr;
    std::vector<uint8_t> pattern;
    size_t offset = 0;
    size_t size = 0;

private:
    FillBuffer(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue) : Command(cmdbuf, queue, CL_COMMAND_FILL_BUFFER) {};
};

struct FillImage : Command
{
    static std::unique_ptr<FillImage> create(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue,
        cl_mem image,
        const void* fill_color,
        const size_t* origin,
        const size_t* region)
    {
        auto ret = std::unique_ptr<FillImage>(
            new FillImage(cmdbuf, queue));

        ret->image = image;

        auto p = reinterpret_cast<const uint8_t*>(fill_color);

        cl_image_format format;
        g_pNextDispatch->clGetImageInfo(
            image,
            CL_IMAGE_FORMAT,
            sizeof(format),
            &format,
            nullptr);

        // The fill color is a single floating-point value for CL_DEPTH images
        // and a 32-bit four-component value for all other image types.
        size_t s =
            format.image_channel_order == CL_DEPTH ?
            sizeof(float) : 4 * sizeof(float);
        ret->fill_color.reserve(s);
        ret->fill_color.insert(
            ret->fill_color.begin(),
            p,
            p + s);

        std::copy(origin, origin + 3, ret->origin.begin());
        std::copy(region, region + 3, ret->region.begin());

        g_pNextDispatch->clRetainMemObject(ret->image);

        return ret;
    }

    ~FillImage()
    {
        g_pNextDispatch->clReleaseMemObject(image);
    }

    int playback(
        cl_command_queue queue,
        std::vector<cl_event>& deps) const override
    {
        auto wait_list = getEventWaitList(deps);
        auto signal = getEventSignalPtr(deps);
        return g_pNextDispatch->clEnqueueFillImage(
            queue,
            image,
            fill_color.data(),
            origin.data(),
            region.data(),
            static_cast<cl_uint>(wait_list.size()),
            wait_list.data(),
            signal);
    }

    cl_mem image = nullptr;
    std::vector<uint8_t> fill_color;
    std::array<size_t, 3> origin = {0, 0, 0};
    std::array<size_t, 3> region = {0, 0, 0};

private:
    FillImage(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue) : Command(cmdbuf, queue, CL_COMMAND_FILL_IMAGE) {};
};

struct NDRangeKernel : Command
{
    static std::unique_ptr<NDRangeKernel> create(
        const cl_ndrange_kernel_command_properties_khr* properties,
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue,
        cl_kernel kernel,
        cl_uint work_dim,
        const size_t* global_work_offset,
        const size_t* global_work_size,
        const size_t* local_work_size,
        cl_int& errorCode)
    {
        errorCode = CL_SUCCESS;

        ptrdiff_t numProperties = 0;
#if defined(cl_khr_command_buffer_mutable_dispatch)
        cl_mutable_dispatch_fields_khr mutableFields = g_MutableDispatchCaps;
#endif

        if( properties )
        {
            const cl_ndrange_kernel_command_properties_khr* check = properties;
            bool found_CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR = false;
            while( errorCode == CL_SUCCESS && check[0] != 0 )
            {
                cl_int  property = (cl_int)check[0];
                switch( property )
                {
#if defined(cl_khr_command_buffer_mutable_dispatch)
                case CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR:
                    if( found_CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR )
                    {
                        errorCode = CL_INVALID_VALUE;
                        return nullptr;
                    }
                    else
                    {
                        found_CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR = true;
                        mutableFields = ((const cl_mutable_dispatch_fields_khr*)(check + 1))[0];
                        check += 2;
                    }
                    break;
#endif
                default:
                    errorCode = CL_INVALID_VALUE;
                    return nullptr;
                }
            }
            numProperties = check - properties + 1;
        }

        auto command = std::unique_ptr<NDRangeKernel>(
            new NDRangeKernel(cmdbuf, queue));

        command->original_kernel = kernel;
        command->kernel = g_pNextDispatch->clCloneKernel(kernel, NULL);
        command->work_dim = work_dim;

#if defined(cl_khr_command_buffer_mutable_dispatch)
        command->mutableFields = mutableFields;
#endif

        command->properties.reserve(numProperties);
        command->properties.insert(
            command->properties.begin(),
            properties,
            properties + numProperties );

        if( global_work_offset )
        {
            command->global_work_offset.reserve(work_dim);
            command->global_work_offset.insert(
                command->global_work_offset.begin(),
                global_work_offset,
                global_work_offset + work_dim);
        }

        if( global_work_size )
        {
            command->global_work_size.reserve(work_dim);
            command->global_work_size.insert(
                command->global_work_size.begin(),
                global_work_size,
                global_work_size + work_dim);
        }

        if( local_work_size )
        {
            command->local_work_size.reserve(work_dim);
            command->local_work_size.insert(
                command->local_work_size.begin(),
                local_work_size,
                local_work_size + work_dim);
        }

        g_pNextDispatch->clRetainKernel(command->original_kernel);

        return command;
    }

    ~NDRangeKernel()
    {
        g_pNextDispatch->clReleaseKernel(kernel);
        g_pNextDispatch->clReleaseKernel(original_kernel);
    }

#if defined(cl_khr_command_buffer_mutable_dispatch)
    cl_int  getInfo(
        cl_mutable_command_info_khr param_name,
        size_t param_value_size,
        void* param_value,
        size_t* param_value_size_ret) override
    {
        switch( param_name )
        {
        case CL_MUTABLE_DISPATCH_PROPERTIES_ARRAY_KHR:
            {
                auto ptr = (cl_ndrange_kernel_command_properties_khr*)param_value;
                return writeVectorToMemory(
                    param_value_size,
                    properties,
                    param_value_size_ret,
                    ptr );
            }
            break;
        case CL_MUTABLE_DISPATCH_KERNEL_KHR:
            {
                auto ptr = (cl_kernel*)param_value;
                return writeParamToMemory(
                    param_value_size,
                    original_kernel,
                    param_value_size_ret,
                    ptr );
            }
            break;
        case CL_MUTABLE_DISPATCH_DIMENSIONS_KHR:
            {
                auto ptr = (cl_uint*)param_value;
                return writeParamToMemory(
                    param_value_size,
                    work_dim,
                    param_value_size_ret,
                    ptr );
            }
            break;
        case CL_MUTABLE_DISPATCH_GLOBAL_WORK_OFFSET_KHR:
            {
                auto ptr = (size_t*)param_value;
                if( global_work_offset.size() )
                {
                    return writeVectorToMemory(
                        param_value_size,
                        global_work_offset,
                        param_value_size_ret,
                        ptr );
                }
                else
                {
                    // TODO: Should it be valid to return a size of zero in
                    // this case instead?
                    std::vector<size_t> temp_global_work_offset(work_dim, 0);
                    return writeVectorToMemory(
                        param_value_size,
                        temp_global_work_offset,
                        param_value_size_ret,
                        ptr );
                }
            }
            break;
        case CL_MUTABLE_DISPATCH_GLOBAL_WORK_SIZE_KHR:
            {
                auto ptr = (size_t*)param_value;
                return writeVectorToMemory(
                    param_value_size,
                    global_work_size,
                    param_value_size_ret,
                    ptr );
            }
        case CL_MUTABLE_DISPATCH_LOCAL_WORK_SIZE_KHR:
            {
                auto ptr = (size_t*)param_value;
                if( local_work_size.size() )
                {
                    return writeVectorToMemory(
                        param_value_size,
                        local_work_size,
                        param_value_size_ret,
                        ptr );
                }
                else
                {
                    // TODO: Should it be valid to return a size of zero in
                    // this case instead?
                    std::vector<size_t> temp_local_work_size(work_dim, 0);
                    return writeVectorToMemory(
                        param_value_size,
                        temp_local_work_size,
                        param_value_size_ret,
                        ptr );
                }
            }
            break;
        default:
            return Command::getInfo(
                param_name,
                param_value_size,
                param_value,
                param_value_size_ret);
        }

        return CL_INVALID_VALUE;
    }

    cl_int  mutate( const cl_mutable_dispatch_config_khr* dispatchConfig )
    {
        //CL_INVALID_OPERATION if values of local_work_size and/or global_work_size result in an increase to the number of work-groups in the ND-range.
        //CL_INVALID_OPERATION if the values of local_work_size and/or global_work_size result in a change to work-group uniformity.
        if( dispatchConfig->work_dim != 0 && dispatchConfig->work_dim != work_dim )
        {
            return CL_INVALID_VALUE;
        }
        if( !(mutableFields & CL_MUTABLE_DISPATCH_GLOBAL_OFFSET_KHR) &&
            dispatchConfig->global_work_offset != nullptr )
        {
            return CL_INVALID_OPERATION;
        }
        if( !(mutableFields & CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR) &&
            dispatchConfig->global_work_size != nullptr )
        {
            return CL_INVALID_OPERATION;
        }
        if( !(mutableFields & CL_MUTABLE_DISPATCH_LOCAL_SIZE_KHR) &&
            dispatchConfig->local_work_size != nullptr )
        {
            return CL_INVALID_OPERATION;
        }
        if( !(mutableFields & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR) &&
            (dispatchConfig->num_args != 0 || dispatchConfig->num_svm_args != 0) )
        {
            return CL_INVALID_OPERATION;
        }
        if( !(mutableFields & CL_MUTABLE_DISPATCH_EXEC_INFO_KHR) &&
            dispatchConfig->num_exec_infos != 0 )
        {
            return CL_INVALID_OPERATION;
        }

        if( ( dispatchConfig->num_args > 0 && dispatchConfig->arg_list == nullptr ) ||
            ( dispatchConfig->num_args == 0 && dispatchConfig->arg_list != nullptr ) )
        {
            return CL_INVALID_VALUE;
        }
        if( ( dispatchConfig->num_svm_args > 0 && dispatchConfig->arg_svm_list == nullptr ) ||
            ( dispatchConfig->num_svm_args == 0 && dispatchConfig->arg_svm_list != nullptr ) )
        {
            return CL_INVALID_VALUE;
        }
        if( ( dispatchConfig->num_exec_infos > 0 && dispatchConfig->exec_info_list == nullptr ) ||
            ( dispatchConfig->num_exec_infos == 0 && dispatchConfig->exec_info_list != nullptr ) )
        {
            return CL_INVALID_VALUE;
        }

        for( cl_uint i = 0; i < dispatchConfig->num_args; i++ )
        {
            if( cl_int errorCode = g_pNextDispatch->clSetKernelArg(
                    kernel,
                    dispatchConfig->arg_list[i].arg_index,
                    dispatchConfig->arg_list[i].arg_size,
                    dispatchConfig->arg_list[i].arg_value ) )
            {
                return errorCode;
            }
        }

        for( cl_uint i = 0; i < dispatchConfig->num_svm_args; i++ )
        {
            if( cl_int errorCode = g_pNextDispatch->clSetKernelArgSVMPointer(
                    kernel,
                    dispatchConfig->arg_svm_list[i].arg_index,
                    dispatchConfig->arg_svm_list[i].arg_value ) )
            {
                return errorCode;
            }
        }

        for( cl_uint i = 0; i < dispatchConfig->num_exec_infos; i++ )
        {
            if( cl_int errorCode = g_pNextDispatch->clSetKernelExecInfo(
                    kernel,
                    dispatchConfig->exec_info_list[i].param_name,
                    dispatchConfig->exec_info_list[i].param_value_size,
                    dispatchConfig->exec_info_list[i].param_value ) )
            {
                return errorCode;
            }
        }

        if( dispatchConfig->global_work_offset )
        {
            global_work_offset.assign(
                dispatchConfig->global_work_offset,
                dispatchConfig->global_work_offset + work_dim );
        }

        if( dispatchConfig->global_work_size )
        {
            global_work_size.assign(
                dispatchConfig->global_work_size,
                dispatchConfig->global_work_size + work_dim );
        }

        if( dispatchConfig->local_work_size )
        {
            local_work_size.assign(
                dispatchConfig->local_work_size,
                dispatchConfig->local_work_size + work_dim );
        }

        return CL_SUCCESS;
    }
#endif // defined(cl_khr_command_buffer_mutable_dispatch)

    int playback(
        cl_command_queue queue,
        std::vector<cl_event>& deps) const override
    {
        auto wait_list = getEventWaitList(deps);
        auto signal = getEventSignalPtr(deps);
        return g_pNextDispatch->clEnqueueNDRangeKernel(
            queue,
            kernel,
            work_dim,
            global_work_offset.size() ? global_work_offset.data() : NULL,
            global_work_size.data(),
            local_work_size.size() ? local_work_size.data() : NULL,
            static_cast<cl_uint>(wait_list.size()),
            wait_list.data(),
            signal);
    }

    cl_kernel original_kernel = nullptr;
    cl_kernel kernel = nullptr;
    cl_uint work_dim = 0;
#if defined(cl_khr_command_buffer_mutable_dispatch)
    cl_mutable_dispatch_fields_khr mutableFields = 0;
#endif
    std::vector<cl_command_buffer_properties_khr> properties;
    std::vector<size_t> global_work_offset;
    std::vector<size_t> global_work_size;
    std::vector<size_t> local_work_size;

private:
    NDRangeKernel(
        cl_command_buffer_khr cmdbuf,
        cl_command_queue queue)
        : Command(cmdbuf, queue, CL_COMMAND_NDRANGE_KERNEL) {};
};

typedef struct _cl_command_buffer_khr
{
    static _cl_command_buffer_khr* create(
        cl_uint num_queues,
        const cl_command_queue* queues,
        const cl_command_buffer_properties_khr* properties,
        cl_int* errcode_ret)
    {
        cl_command_buffer_khr cmdbuf = NULL;
        cl_int errorCode = CL_SUCCESS;

        ptrdiff_t numProperties = 0;
        cl_command_buffer_flags_khr flags = 0;

        if( num_queues != 1 || queues == NULL )
        {
            errorCode = CL_INVALID_VALUE;
        }
        if( properties )
        {
            const cl_command_buffer_properties_khr* check = properties;
            bool found_CL_COMMAND_BUFFER_FLAGS_KHR = false;
            while( errorCode == CL_SUCCESS && check[0] != 0 )
            {
                cl_int  property = (cl_int)check[0];
                switch( property )
                {
                case CL_COMMAND_BUFFER_FLAGS_KHR:
                    if( found_CL_COMMAND_BUFFER_FLAGS_KHR )
                    {
                        errorCode = CL_INVALID_VALUE;
                    }
                    else
                    {
                        found_CL_COMMAND_BUFFER_FLAGS_KHR = true;
                        flags = ((const cl_command_buffer_flags_khr*)(check + 1))[0];
                        check += 2;
                    }
                    break;
                default:
                    errorCode = CL_INVALID_VALUE;
                    break;
                }
            }
            numProperties = check - properties + 1;
        }
        if( errcode_ret )
        {
            errcode_ret[0] = errorCode;
        }
        if( errorCode == CL_SUCCESS) {
            cmdbuf = new _cl_command_buffer_khr(flags);
            cmdbuf->Queues.reserve(num_queues);
            cmdbuf->Queues.insert(
                cmdbuf->Queues.begin(),
                queues,
                queues + num_queues );
            cmdbuf->Properties.reserve(numProperties);
            cmdbuf->Properties.insert(
                cmdbuf->Properties.begin(),
                properties,
                properties + numProperties );

            for( auto queue : cmdbuf->Queues )
            {
                g_pNextDispatch->clRetainCommandQueue(queue);
            }
        }

        return cmdbuf;
    }

    ~_cl_command_buffer_khr()
    {
        for( auto queue : Queues )
        {
            g_pNextDispatch->clReleaseCommandQueue(queue);
        }
    }

    static bool isValid( cl_command_buffer_khr cmdbuf )
    {
        return cmdbuf && cmdbuf->Magic == cMagic;
    }

    cl_int  retain()
    {
        RefCount.fetch_add(1, std::memory_order_relaxed);
        return CL_SUCCESS;
    }

    cl_int  release()
    {
        RefCount.fetch_sub(1, std::memory_order_relaxed);
        if( RefCount.load(std::memory_order_relaxed) == 0 )
        {
            delete this;
        }
        return CL_SUCCESS;
    }

    cl_command_queue    getQueue() const
    {
        return Queues[0];
    }

    cl_int  getInfo(
                cl_command_buffer_info_khr param_name,
                size_t param_value_size,
                void* param_value,
                size_t* param_value_size_ret)
    {
        switch( param_name )
        {
        case CL_COMMAND_BUFFER_QUEUES_KHR:
            {
                auto ptr = (cl_command_queue*)param_value;
                return writeVectorToMemory(
                    param_value_size,
                    Queues,
                    param_value_size_ret,
                    ptr );
            }
            break;
        case CL_COMMAND_BUFFER_NUM_QUEUES_KHR:
            {
                auto ptr = (cl_uint*)param_value;
                return writeParamToMemory(
                    param_value_size,
                    static_cast<cl_uint>(Queues.size()),
                    param_value_size_ret,
                    ptr );
            }
            break;
        case CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR:
            {
                auto ptr = (cl_uint*)param_value;
                return writeParamToMemory(
                    param_value_size,
                    RefCount.load(std::memory_order_relaxed),
                    param_value_size_ret,
                    ptr );
            }
            break;
        case CL_COMMAND_BUFFER_STATE_KHR:
            {
                auto ptr = (cl_command_buffer_state_khr*)param_value;
                return writeParamToMemory(
                    param_value_size,
                    State,
                    param_value_size_ret,
                    ptr );
            }
            break;
        case CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR:
            {
                auto ptr = (cl_command_buffer_properties_khr*)param_value;
                return writeVectorToMemory(
                    param_value_size,
                    Properties,
                    param_value_size_ret,
                    ptr );
            }
            break;
#if defined(CL_COMMAND_BUFFER_CONTEXT_KHR)
        // This enum was not in the original specification so some headers may
        // not have it.
        case CL_COMMAND_BUFFER_CONTEXT_KHR:
            {
                cl_context context = nullptr;
                if( cl_int errorCode = g_pNextDispatch->clGetCommandQueueInfo(
                        getQueue(),
                        CL_QUEUE_CONTEXT,
                        sizeof(context),
                        &context,
                        nullptr) )
                {
                    return errorCode;
                }
                auto ptr = (cl_context*)param_value;
                return writeParamToMemory(
                    param_value_size,
                    context,
                    param_value_size_ret,
                    ptr );
            }
            break;
#endif
        default:
            break;
        }

        return CL_INVALID_VALUE;
    }

    cl_int  checkRecordErrors(
                cl_command_queue queue,
                cl_uint num_sync_points_in_wait_list,
                const cl_sync_point_khr* sync_point_wait_list,
                cl_mutable_command_khr* mutable_handle )
    {
        if( State != CL_COMMAND_BUFFER_STATE_RECORDING_KHR )
        {
            return CL_INVALID_OPERATION;
        }
        if( queue != NULL )
        {
            return CL_INVALID_COMMAND_QUEUE;
        }
// TODO: Change this to a runtime check?
#if !defined(cl_khr_command_buffer_mutable_dispatch)
        if( mutable_handle != NULL )
        {
            return CL_INVALID_VALUE;
        }
#endif // !defined(cl_khr_command_buffer_mutable_dispatch)
        if( ( sync_point_wait_list == NULL && num_sync_points_in_wait_list > 0 ) ||
            ( sync_point_wait_list != NULL && num_sync_points_in_wait_list == 0 ) )
        {
            return CL_INVALID_SYNC_POINT_WAIT_LIST_KHR;
        }

        uint32_t numSyncPoints = NextSyncPoint.load(std::memory_order_relaxed);
        for( cl_uint i = 0; i < num_sync_points_in_wait_list; i++ )
        {
            if( sync_point_wait_list[i] == 0 || sync_point_wait_list[i] >= numSyncPoints )
            {
                return CL_INVALID_SYNC_POINT_WAIT_LIST_KHR;
            }
        }

        // CL_INVALID_CONTEXT if queue and cmdbuf do not have the same context?

        return CL_SUCCESS;
    }

    cl_int  checkPlaybackErrors(
                cl_uint num_queues,
                cl_command_queue* queues,
                cl_uint num_events_in_wait_list,
                const cl_event* event_wait_list )
    {
        if( State != CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR )
        {
            return CL_INVALID_OPERATION;
        }
        if( ( queues == NULL && num_queues > 0 ) ||
            ( queues != NULL && num_queues == 0 ) )
        {
            return CL_INVALID_VALUE;
        }
        if( num_queues > 1 )
        {
            return CL_INVALID_VALUE;
        }

        // CL_INCOMPATIBLE_COMMAND_QUEUE_KHR if any element of queues is not compatible with the command-queue set on command_buffer creation at the same list index.
        // CL_INVALID_CONTEXT if any element of queues does not have the same context as the command-queue set on command_buffer creation at the same list indes.
        // CL_INVALID_CONTEXT if the context associated with the command buffer and events in event_wait_list are not the same.

        return CL_SUCCESS;
    }

    void    addCommand(
                std::unique_ptr<Command> command,
                cl_uint num_sync_points,
                const cl_sync_point_khr* wait_list,
                cl_sync_point_khr* sync_point,
                cl_mutable_command_khr* mutable_handle )
    {
        cl_sync_point_khr syncPoint =
            sync_point != nullptr ?
            NextSyncPoint.fetch_add(1, std::memory_order_relaxed) :
            0;

        command->addDependencies(
            num_sync_points,
            wait_list,
            syncPoint);

        if( sync_point != nullptr )
        {
            sync_point[0] = syncPoint;
        }
        if( mutable_handle != nullptr )
        {
            mutable_handle[0] = command.get();
        }

        Commands.push_back(std::move(command));
    }

    cl_int  finalize()
    {
        if( State != CL_COMMAND_BUFFER_STATE_RECORDING_KHR )
        {
            return CL_INVALID_OPERATION;
        }

        State = CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR;
        return CL_SUCCESS;
    }

    cl_int  replay(
                cl_command_queue queue) const
    {
        cl_int errorCode = CL_SUCCESS;

        const uint32_t numSyncPoints = NextSyncPoint.load(std::memory_order_relaxed);
        std::vector<cl_event> deps(numSyncPoints, nullptr);

        for( const auto& command : Commands )
        {
            errorCode = command->playback(queue, deps);
            if( errorCode != CL_SUCCESS )
            {
                break;
            }
        }

        for( auto event : deps )
        {
            if (event != nullptr)
            {
                g_pNextDispatch->clReleaseEvent(event);
            }
        }

        return errorCode;
    }

#if defined(cl_khr_command_buffer_mutable_dispatch)
    cl_int  mutate( const cl_mutable_base_config_khr* mutable_config )
    {
        if( State != CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR )
        {
            return CL_INVALID_OPERATION;
        }
        if( !(Flags & CL_COMMAND_BUFFER_MUTABLE_KHR) )
        {
            return CL_INVALID_OPERATION;
        }

        if( mutable_config == nullptr )
        {
            return CL_INVALID_VALUE;
        }
        if( mutable_config->type != CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR )
        {
            return CL_INVALID_VALUE;
        }
        if( mutable_config->next == nullptr && mutable_config->mutable_dispatch_list == nullptr )
        {
            return CL_INVALID_VALUE;
        }
        if( ( mutable_config->num_mutable_dispatch > 0 && mutable_config->mutable_dispatch_list == nullptr ) ||
            ( mutable_config->num_mutable_dispatch == 0 && mutable_config->mutable_dispatch_list != nullptr ) )
        {
            return CL_INVALID_VALUE;
        }
        // No "next" extensions are currently supported.
        if( mutable_config->next != nullptr )
        {
            return CL_INVALID_VALUE;
        }

        for( cl_uint i = 0; i < mutable_config->num_mutable_dispatch; i++ )
        {
            const cl_mutable_dispatch_config_khr* dispatchConfig =
                &mutable_config->mutable_dispatch_list[i];
            if( !Command::isValid(dispatchConfig->command) ||
                dispatchConfig->command->getCmdBuf() != this )
            {
                return CL_INVALID_MUTABLE_COMMAND_KHR;
            }
            if( dispatchConfig->type == CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR )
            {
                if( dispatchConfig->command->getType() != CL_COMMAND_NDRANGE_KERNEL )
                {
                    return CL_INVALID_MUTABLE_COMMAND_KHR;
                }
                
                if( cl_int errorCode = ((NDRangeKernel*)dispatchConfig->command)->mutate(
                        dispatchConfig ) )
                {
                    return errorCode;
                }
            }
            else
            {
                return CL_INVALID_VALUE;
            }
        }

        return CL_SUCCESS;
    }
#endif // defined(cl_khr_command_buffer_mutable_dispatch)

private:
    static constexpr cl_uint cMagic = 0x434d4442;   // "CMDB"

    const cl_uint Magic;
    std::vector<cl_command_queue>   Queues;
    std::vector<cl_command_buffer_properties_khr>   Properties;
    cl_command_buffer_state_khr State;
    cl_command_buffer_flags_khr Flags;
    std::atomic<uint32_t> RefCount;

    std::vector<std::unique_ptr<Command>> Commands;
    std::atomic<uint32_t> NextSyncPoint;

    _cl_command_buffer_khr(cl_command_buffer_flags_khr flags) :
        Magic(cMagic),
        State(CL_COMMAND_BUFFER_STATE_RECORDING_KHR),
        Flags(flags),
        RefCount(1),
        NextSyncPoint(1) {}
} CommandBuffer;

///////////////////////////////////////////////////////////////////////////////
//
// We need to define the mutable command constructor separately and after the
// definition of a command buffer because we will call into the command buffer
// to get the queue if the passed-in queue is NULL.
_cl_mutable_command_khr::_cl_mutable_command_khr(
    cl_command_buffer_khr cmdbuf,
    cl_command_queue queue,
    cl_command_type type) :
    Magic(cMagic),
    Type(type),
    CmdBuf(cmdbuf),
    Queue(queue ? queue : cmdbuf->getQueue()) {}

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer
cl_command_buffer_khr CL_API_CALL clCreateCommandBufferKHR_EMU(
    cl_uint num_queues,
    const cl_command_queue* queues,
    const cl_command_buffer_properties_khr* properties,
    cl_int* errcode_ret)
{
    return CommandBuffer::create(
        num_queues,
        queues,
        properties,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer
cl_int CL_API_CALL clFinalizeCommandBufferKHR_EMU(
    cl_command_buffer_khr cmdbuf)
{
    if( !CommandBuffer::isValid(cmdbuf) )
    {
        return CL_INVALID_COMMAND_BUFFER_KHR;
    }

    return cmdbuf->finalize();
}

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer
cl_int CL_API_CALL clRetainCommandBufferKHR_EMU(
    cl_command_buffer_khr cmdbuf)
{
    if( !CommandBuffer::isValid(cmdbuf) )
    {
        return CL_INVALID_COMMAND_BUFFER_KHR;
    }

    return cmdbuf->retain();
}

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer
cl_int CL_API_CALL clReleaseCommandBufferKHR_EMU(
    cl_command_buffer_khr cmdbuf)
{
    if( !CommandBuffer::isValid(cmdbuf) )
    {
        return CL_INVALID_COMMAND_BUFFER_KHR;
    }

    return cmdbuf->release();
}

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer
cl_int CL_API_CALL clEnqueueCommandBufferKHR_EMU(
    cl_uint num_queues,
    cl_command_queue* queues,
    cl_command_buffer_khr cmdbuf,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    if( !CommandBuffer::isValid(cmdbuf) )
    {
        return CL_INVALID_COMMAND_BUFFER_KHR;
    }
    if( cl_int errorCode = cmdbuf->checkPlaybackErrors(
            num_queues,
            queues,
            num_events_in_wait_list,
            event_wait_list) )
    {
        return errorCode;
    }

    cl_command_queue queue = cmdbuf->getQueue();
    if( num_queues > 0 )
    {
        queue = queues[0];
    }

    cl_int errorCode = CL_SUCCESS;
    cl_event startEvent = nullptr;

    if( num_events_in_wait_list || event )
    {
        errorCode = g_pNextDispatch->clEnqueueBarrierWithWaitList(
            queue,
            num_events_in_wait_list,
            event_wait_list,
            event ? &startEvent : nullptr);
    }

    if( errorCode == CL_SUCCESS )
    {
        errorCode = cmdbuf->replay(queue);
    }

    if( errorCode == CL_SUCCESS && event )
    {
        errorCode = g_pNextDispatch->clEnqueueBarrierWithWaitList(
            queue,
            0,
            NULL,
            event );
    }

    if( event )
    {
        if( errorCode == CL_SUCCESS )
        {
            getLayerContext().EventMap[event[0]] = startEvent;
        }
        else
        {
            g_pNextDispatch->clReleaseEvent(startEvent);
        }
    }

    return errorCode;
}

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer
cl_int CL_API_CALL clCommandBarrierWithWaitListKHR_EMU(
    cl_command_buffer_khr cmdbuf,
    cl_command_queue command_queue,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle)
{
    if( !CommandBuffer::isValid(cmdbuf) )
    {
        return CL_INVALID_COMMAND_BUFFER_KHR;
    }
    if( cl_int errorCode = cmdbuf->checkRecordErrors(
            command_queue,
            num_sync_points_in_wait_list,
            sync_point_wait_list,
            mutable_handle) )
    {
        return errorCode;
    }

    cmdbuf->addCommand(
        BarrierWithWaitList::create(cmdbuf, command_queue),
        num_sync_points_in_wait_list,
        sync_point_wait_list,
        sync_point,
        mutable_handle);
    return CL_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer
cl_int CL_API_CALL clCommandCopyBufferKHR_EMU(
    cl_command_buffer_khr cmdbuf,
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    size_t src_offset,
    size_t dst_offset,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle)
{
    if( !CommandBuffer::isValid(cmdbuf) )
    {
        return CL_INVALID_COMMAND_BUFFER_KHR;
    }
    if( cl_int errorCode = cmdbuf->checkRecordErrors(
            command_queue,
            num_sync_points_in_wait_list,
            sync_point_wait_list,
            mutable_handle) )
    {
        return errorCode;
    }

    cmdbuf->addCommand(
        CopyBuffer::create(
            cmdbuf,
            command_queue,
            src_buffer,
            dst_buffer,
            src_offset,
            dst_offset,
            size),
        num_sync_points_in_wait_list,
        sync_point_wait_list,
        sync_point,
        mutable_handle);
    return CL_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer
cl_int CL_API_CALL clCommandCopyBufferRectKHR_EMU(
    cl_command_buffer_khr cmdbuf,
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle)
{
    if( !CommandBuffer::isValid(cmdbuf) )
    {
        return CL_INVALID_COMMAND_BUFFER_KHR;
    }
    if( cl_int errorCode = cmdbuf->checkRecordErrors(
            command_queue,
            num_sync_points_in_wait_list,
            sync_point_wait_list,
            mutable_handle) )
    {
        return errorCode;
    }

    cmdbuf->addCommand(
        CopyBufferRect::create(
            cmdbuf,
            command_queue,
            src_buffer,
            dst_buffer,
            src_origin,
            dst_origin,
            region,
            src_row_pitch,
            src_slice_pitch,
            dst_row_pitch,
            dst_slice_pitch),
        num_sync_points_in_wait_list,
        sync_point_wait_list,
        sync_point,
        mutable_handle);
    return CL_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer
cl_int CL_API_CALL clCommandCopyBufferToImageKHR_EMU(
    cl_command_buffer_khr cmdbuf,
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_image,
    size_t src_offset,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle)
{
    if( !CommandBuffer::isValid(cmdbuf) )
    {
        return CL_INVALID_COMMAND_BUFFER_KHR;
    }
    if( cl_int errorCode = cmdbuf->checkRecordErrors(
            command_queue,
            num_sync_points_in_wait_list,
            sync_point_wait_list,
            mutable_handle) )
    {
        return errorCode;
    }

    cmdbuf->addCommand(
        CopyBufferToImage::create(
            cmdbuf,
            command_queue,
            src_buffer,
            dst_image,
            src_offset,
            dst_origin,
            region),
        num_sync_points_in_wait_list,
        sync_point_wait_list,
        sync_point,
        mutable_handle);
    return CL_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer
cl_int CL_API_CALL clCommandCopyImageKHR_EMU(
    cl_command_buffer_khr cmdbuf,
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_image,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle)
{
    if( !CommandBuffer::isValid(cmdbuf) )
    {
        return CL_INVALID_COMMAND_BUFFER_KHR;
    }
    if( cl_int errorCode = cmdbuf->checkRecordErrors(
            command_queue,
            num_sync_points_in_wait_list,
            sync_point_wait_list,
            mutable_handle) )
    {
        return errorCode;
    }

    cmdbuf->addCommand(
        CopyImage::create(
            cmdbuf,
            command_queue,
            src_image,
            dst_image,
            src_origin,
            dst_origin,
            region),
        num_sync_points_in_wait_list,
        sync_point_wait_list,
        sync_point,
        mutable_handle);
    return CL_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer
cl_int CL_API_CALL clCommandCopyImageToBufferKHR_EMU(
    cl_command_buffer_khr cmdbuf,
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* region,
    size_t dst_offset,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle)
{
    if( !CommandBuffer::isValid(cmdbuf) )
    {
        return CL_INVALID_COMMAND_BUFFER_KHR;
    }
    if( cl_int errorCode = cmdbuf->checkRecordErrors(
            command_queue,
            num_sync_points_in_wait_list,
            sync_point_wait_list,
            mutable_handle) )
    {
        return errorCode;
    }

    cmdbuf->addCommand(
        CopyImageToBuffer::create(
            cmdbuf,
            command_queue,
            src_image,
            dst_buffer,
            src_origin,
            region,
            dst_offset),
        num_sync_points_in_wait_list,
        sync_point_wait_list,
        sync_point,
        mutable_handle);
    return CL_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer
cl_int CL_API_CALL clCommandFillBufferKHR_EMU(
    cl_command_buffer_khr cmdbuf,
    cl_command_queue command_queue,
    cl_mem buffer,
    const void* pattern,
    size_t pattern_size,
    size_t offset,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle)
{
    if( !CommandBuffer::isValid(cmdbuf) )
    {
        return CL_INVALID_COMMAND_BUFFER_KHR;
    }
    if( cl_int errorCode = cmdbuf->checkRecordErrors(
            command_queue,
            num_sync_points_in_wait_list,
            sync_point_wait_list,
            mutable_handle) )
    {
        return errorCode;
    }

    cmdbuf->addCommand(
        FillBuffer::create(
            cmdbuf,
            command_queue,
            buffer,
            pattern,
            pattern_size,
            offset,
            size),
        num_sync_points_in_wait_list,
        sync_point_wait_list,
        sync_point,
        mutable_handle);
    return CL_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer
cl_int CL_API_CALL clCommandFillImageKHR_EMU(
    cl_command_buffer_khr cmdbuf,
    cl_command_queue command_queue,
    cl_mem image,
    const void* fill_color,
    const size_t* origin,
    const size_t* region,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle)
{
    if( !CommandBuffer::isValid(cmdbuf) )
    {
        return CL_INVALID_COMMAND_BUFFER_KHR;
    }
    if( cl_int errorCode = cmdbuf->checkRecordErrors(
            command_queue,
            num_sync_points_in_wait_list,
            sync_point_wait_list,
            mutable_handle) )
    {
        return errorCode;
    }

    cmdbuf->addCommand(
        FillImage::create(
            cmdbuf,
            command_queue,
            image,
            fill_color,
            origin,
            region),
        num_sync_points_in_wait_list,
        sync_point_wait_list,
        sync_point,
        mutable_handle);
    return CL_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer
cl_int CL_API_CALL clCommandNDRangeKernelKHR_EMU(
    cl_command_buffer_khr cmdbuf,
    cl_command_queue command_queue,
    const cl_ndrange_kernel_command_properties_khr* properties,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* global_work_size,
    const size_t* local_work_size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle)
{
    if( !CommandBuffer::isValid(cmdbuf) )
    {
        return CL_INVALID_COMMAND_BUFFER_KHR;
    }
    if( cl_int errorCode = cmdbuf->checkRecordErrors(
            command_queue,
            num_sync_points_in_wait_list,
            sync_point_wait_list,
            mutable_handle) )
    {
        return errorCode;
    }

    cl_int errorCode = CL_SUCCESS;
    auto command = NDRangeKernel::create(
        properties,
        cmdbuf,
        command_queue,
        kernel,
        work_dim,
        global_work_offset,
        global_work_size,
        local_work_size,
        errorCode);
    if( errorCode )
    {
        return errorCode;
    }

    cmdbuf->addCommand(
        std::move(command),
        num_sync_points_in_wait_list,
        sync_point_wait_list,
        sync_point,
        mutable_handle);
    return CL_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer
cl_int CL_API_CALL clGetCommandBufferInfoKHR_EMU(
    cl_command_buffer_khr cmdbuf,
    cl_command_buffer_info_khr param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if( !CommandBuffer::isValid(cmdbuf) )
    {
        return CL_INVALID_COMMAND_BUFFER_KHR;
    }

    return cmdbuf->getInfo(
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#if defined(cl_khr_command_buffer_multi_device) && 0

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer_multi_device
cl_command_buffer_khr CL_API_CALL clRemapCommandBufferKHR_EMU(
    cl_command_buffer_khr command_buffer,
    cl_bool automatic,
    cl_uint num_queues,
    const cl_command_queue* queues,
    cl_uint num_handles,
    const cl_mutable_command_khr* handles,
    cl_mutable_command_khr* handles_ret,
    cl_int* errcode_ret)
{
    if (errcode_ret) {
        errcode_ret[0] = CL_OUT_OF_HOST_MEMORY;
    }
    return nullptr;
}

#endif

#if defined(cl_khr_command_buffer_mutable_dispatch)

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer_mutable_dispatch
cl_int CL_API_CALL clUpdateMutableCommandsKHR_EMU(
    cl_command_buffer_khr cmdbuf,
    const cl_mutable_base_config_khr* mutable_config)
{
    if( !CommandBuffer::isValid(cmdbuf) )
    {
        return CL_INVALID_COMMAND_BUFFER_KHR;
    }
    if( cl_int errorCode = cmdbuf->mutate(
            mutable_config ) )
    {
        return errorCode;
    }

    return CL_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
//
// cl_khr_command_buffer_mutable_dispatch
cl_int CL_API_CALL clGetMutableCommandInfoKHR_EMU(
    cl_mutable_command_khr command,
    cl_mutable_command_info_khr param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if( !Command::isValid(command) )
    {
        return CL_INVALID_MUTABLE_COMMAND_KHR;
    }

    return command->getInfo(
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#endif // defined(cl_khr_command_buffer_mutable_dispatch)

bool clGetDeviceInfo_override(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret)
{
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

            size = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_VERSION,
                0,
                nullptr,
                &size );

            std::vector<char> deviceVersion(size);
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_VERSION,
                size,
                deviceVersion.data(),
                nullptr );

            if( checkStringForExtension(
                    deviceExtensions.data(),
                    CL_KHR_COMMAND_BUFFER_EXTENSION_NAME ) == false &&
                getOpenCLVersionFromString(
                    deviceVersion.data() ) >= 0x00020001)
            {
                std::string newExtensions;
                newExtensions += CL_KHR_COMMAND_BUFFER_EXTENSION_NAME;
#if defined(cl_khr_command_buffer_mutable_dispatch)
                newExtensions += ' ';
                newExtensions += CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_EXTENSION_NAME;
#endif

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

            size = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_VERSION,
                0,
                nullptr,
                &size );

            std::vector<char> deviceVersion(size);
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_VERSION,
                size,
                deviceVersion.data(),
                nullptr );

            bool found = false;
            for( const auto& extension : extensions )
            {
                if( strcmp(extension.name, CL_KHR_COMMAND_BUFFER_EXTENSION_NAME) == 0 )
                {
                    found = true;
                    break;
                }
            }

            if( found == false &&
                getOpenCLVersionFromString(
                    deviceVersion.data() ) >= 0x00020001)
            {
                {
                    extensions.emplace_back();
                    cl_name_version& extension = extensions.back();

                    memset(extension.name, 0, CL_NAME_VERSION_MAX_NAME_SIZE);
                    strcpy(extension.name, CL_KHR_COMMAND_BUFFER_EXTENSION_NAME);

                    extension.version = version_cl_khr_command_buffer;
                }
#if defined(cl_khr_command_buffer_mutable_dispatch)
                {
                    extensions.emplace_back();
                    cl_name_version& extension = extensions.back();

                    memset(extension.name, 0, CL_NAME_VERSION_MAX_NAME_SIZE);
                    strcpy(extension.name, CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_EXTENSION_NAME);

                    extension.version = version_cl_khr_command_buffer_mutable_dispatch;
                }
#endif

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
    case CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR:
        {
            cl_device_command_buffer_capabilities_khr caps =
                CL_COMMAND_BUFFER_CAPABILITY_KERNEL_PRINTF_KHR |
                CL_COMMAND_BUFFER_CAPABILITY_SIMULTANEOUS_USE_KHR;

            cl_device_device_enqueue_capabilities dseCaps = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES,
                sizeof(dseCaps),
                &dseCaps,
                nullptr );
            if( dseCaps != 0 )
            {
                caps |= CL_COMMAND_BUFFER_CAPABILITY_DEVICE_SIDE_ENQUEUE_KHR;
            }

            cl_command_queue_properties cqProps = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_QUEUE_PROPERTIES,
                sizeof(cqProps),
                &cqProps,
                nullptr );
            if( cqProps & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE )
            {
                caps |= CL_COMMAND_BUFFER_CAPABILITY_OUT_OF_ORDER_KHR;
            }

            auto ptr = (cl_device_command_buffer_capabilities_khr*)param_value;
            cl_int errorCode = writeParamToMemory(
                param_value_size,
                caps,
                param_value_size_ret,
                ptr );

            if( errcode_ret )
            {
                errcode_ret[0] = errorCode;
            }
            return true;
        }
        break;
    case CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR:
        {
            // No properties are currently required.
            cl_command_queue_properties props = 0;

            auto ptr = (cl_command_queue_properties*)param_value;
            cl_int errorCode = writeParamToMemory(
                param_value_size,
                props,
                param_value_size_ret,
                ptr );

            if( errcode_ret )
            {
                errcode_ret[0] = errorCode;
            }
            return true;
        }
        break;
#if defined(cl_khr_command_buffer_mutable_dispatch)
    case CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR:
        {
            cl_mutable_dispatch_fields_khr caps =
                g_MutableDispatchCaps;

            auto ptr = (cl_mutable_dispatch_fields_khr*)param_value;
            cl_int errorCode = writeParamToMemory(
                param_value_size,
                caps,
                param_value_size_ret,
                ptr );

            if( errcode_ret )
            {
                errcode_ret[0] = errorCode;
            }
            return true;
        }
        break;
#endif
    default: break;
    }

    return false;
}

bool clGetEventInfo_override(
    cl_event event,
    cl_event_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret)
{
    switch(param_name) {
    case CL_EVENT_COMMAND_TYPE:
        {
            auto& context = getLayerContext();
            auto it = context.EventMap.find(event);
            if (it != context.EventMap.end()) {
                cl_command_type type = CL_COMMAND_COMMAND_BUFFER_KHR;
                auto ptr = (cl_command_type*)param_value;
                cl_int errorCode = writeParamToMemory(
                    param_value_size,
                    type,
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

bool clGetEventProfilingInfo_override(
    cl_event event,
    cl_profiling_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret)
{
    switch(param_name) {
    case CL_PROFILING_COMMAND_QUEUED:
    case CL_PROFILING_COMMAND_SUBMIT:
    case CL_PROFILING_COMMAND_START:
        {
            auto& context = getLayerContext();
            auto it = context.EventMap.find(event);
            if (it != context.EventMap.end()) {
                cl_int errorCode = g_pNextDispatch->clGetEventProfilingInfo(
                    it->second,
                    param_name,
                    param_value_size,
                    param_value,
                    param_value_size_ret);
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

            size = 0;
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_VERSION,
                0,
                nullptr,
                &size );

            std::vector<char> platformVersion(size);
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_VERSION,
                size,
                platformVersion.data(),
                nullptr );

            if( checkStringForExtension(
                    platformExtensions.data(),
                    CL_KHR_COMMAND_BUFFER_EXTENSION_NAME ) == false &&
                getOpenCLVersionFromString(
                    platformVersion.data() ) >= 0x00020001)
            {
                std::string newExtensions;
                newExtensions += CL_KHR_COMMAND_BUFFER_EXTENSION_NAME;
#if defined(cl_khr_command_buffer_mutable_dispatch)
                newExtensions += ' ';
                newExtensions += CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_EXTENSION_NAME;
#endif

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

            size = 0;
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_VERSION,
                0,
                nullptr,
                &size );

            std::vector<char> platformVersion(size);
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_VERSION,
                size,
                platformVersion.data(),
                nullptr );

            bool found = false;
            for( const auto& extension : extensions )
            {
                if( strcmp(extension.name, CL_KHR_COMMAND_BUFFER_EXTENSION_NAME) == 0 )
                {
                    found = true;
                    break;
                }
            }

            if( found == false &&
                getOpenCLVersionFromString(
                    platformVersion.data() ) >= 0x00020001)
            {
                {
                    extensions.emplace_back();
                    cl_name_version& extension = extensions.back();

                    memset(extension.name, 0, CL_NAME_VERSION_MAX_NAME_SIZE);
                    strcpy(extension.name, CL_KHR_COMMAND_BUFFER_EXTENSION_NAME);

                    extension.version = version_cl_khr_command_buffer;
                }
#if defined(cl_khr_command_buffer_mutable_dispatch)
                {
                    extensions.emplace_back();
                    cl_name_version& extension = extensions.back();

                    memset(extension.name, 0, CL_NAME_VERSION_MAX_NAME_SIZE);
                    strcpy(extension.name, CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_EXTENSION_NAME);

                    extension.version = version_cl_khr_command_buffer_mutable_dispatch;
                }
#endif

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
