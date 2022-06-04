/*
// Copyright (c) 20220 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_layer.h>

#include <array>
#include <atomic>
#include <vector>

#include "layer_util.hpp"

extern const struct _cl_icd_dispatch* g_pNextDispatch;

namespace CmdBuf {

struct Command
{
    virtual ~Command() = default;
    virtual int playback(
        cl_command_queue) = 0;
};

struct BarrierWithWaitList : Command
{
    static BarrierWithWaitList* create(void)
    {
        auto* ret = new BarrierWithWaitList();
        return ret;
    }

    int playback(
        cl_command_queue queue) override
    {
        return g_pNextDispatch->clEnqueueBarrierWithWaitList(
            queue,
            0,
            NULL,
            0);
    }

private:
    BarrierWithWaitList() = default;
};

struct CopyBuffer : Command
{
    static CopyBuffer* create(
        cl_mem src_buffer,
        cl_mem dst_buffer,
        size_t src_offset,
        size_t dst_offset,
        size_t size)
    {
        auto* ret = new CopyBuffer();

        ret->src_buffer = src_buffer;
        ret->dst_buffer = dst_buffer;
        ret->src_offset = src_offset;
        ret->dst_offset = dst_offset;
        ret->size = size;

        return ret;
    }

    int playback(
        cl_command_queue queue) override
    {
        return g_pNextDispatch->clEnqueueCopyBuffer(
            queue,
            src_buffer,
            dst_buffer,
            src_offset,
            dst_offset,
            size,
            0,
            NULL,
            0);
    }

    cl_mem src_buffer = nullptr;
    cl_mem dst_buffer = nullptr;
    size_t src_offset = 0;
    size_t dst_offset = 0;
    size_t size = 0;

private:
    CopyBuffer() = default;
};

struct CopyBufferRect : Command
{
    static CopyBufferRect* create(
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
        auto* ret = new CopyBufferRect();

        ret->src_buffer = src_buffer;
        ret->dst_buffer = dst_buffer;
        std::copy(src_origin, src_origin + 3, ret->src_origin.begin());
        std::copy(dst_origin, dst_origin + 3, ret->dst_origin.begin());
        std::copy(region, region + 3, ret->region.begin());
        ret->src_row_pitch = src_row_pitch;
        ret->src_slice_pitch = src_slice_pitch;
        ret->dst_row_pitch = dst_row_pitch;
        ret->dst_slice_pitch = dst_slice_pitch;

        return ret;
    }

    int playback(
        cl_command_queue queue) override
    {
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
            0,
            NULL,
            0);
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
    CopyBufferRect() = default;
};

struct CopyBufferToImage : Command
{
    static CopyBufferToImage* create(
        cl_mem src_buffer,
        cl_mem dst_image,
        size_t src_offset,
        const size_t* dst_origin,
        const size_t* region)
    {
        auto* ret = new CopyBufferToImage();

        ret->src_buffer = src_buffer;
        ret->dst_image = dst_image;
        ret->src_offset = src_offset;
        std::copy(dst_origin, dst_origin + 3, ret->dst_origin.begin());
        std::copy(region, region + 3, ret->region.begin());

        return ret;
    }

    int playback(
        cl_command_queue queue) override
    {
        return g_pNextDispatch->clEnqueueCopyBufferToImage(
            queue,
            src_buffer,
            dst_image,
            src_offset,
            dst_origin.data(),
            region.data(),
            0,
            NULL,
            0);
    }

    cl_mem src_buffer = nullptr;
    cl_mem dst_image = nullptr;
    size_t src_offset = 0;
    std::array<size_t, 3> dst_origin = {0, 0, 0};
    std::array<size_t, 3> region = {0, 0, 0};

private:
    CopyBufferToImage() = default;
};

struct CopyImage : Command
{
    static CopyImage* create(
        cl_mem src_image,
        cl_mem dst_image,
        const size_t* src_origin,
        const size_t* dst_origin,
        const size_t* region)
    {
        auto* ret = new CopyImage();

        ret->src_image = src_image;
        ret->dst_image = dst_image;
        std::copy(src_origin, src_origin + 3, ret->src_origin.begin());
        std::copy(dst_origin, dst_origin + 3, ret->dst_origin.begin());
        std::copy(region, region + 3, ret->region.begin());

        return ret;
    }

    int playback(
        cl_command_queue queue) override
    {
        return g_pNextDispatch->clEnqueueBarrierWithWaitList(
            queue,
            0,
            NULL,
            0);
    }

    cl_mem src_image = nullptr;
    cl_mem dst_image = nullptr;
    std::array<size_t, 3> src_origin = {0, 0, 0};
    std::array<size_t, 3> dst_origin = {0, 0, 0};
    std::array<size_t, 3> region = {0, 0, 0};

private:
    CopyImage() = default;
};

struct CopyImageToBuffer : Command
{
    static CopyImageToBuffer* create(
        cl_mem src_image,
        cl_mem dst_buffer,
        const size_t* src_origin,
        const size_t* region,
        size_t dst_offset)
    {
        auto* ret = new CopyImageToBuffer();

        ret->src_image = src_image;
        ret->dst_buffer = dst_buffer;
        std::copy(src_origin, src_origin + 3, ret->src_origin.begin());
        std::copy(region, region + 3, ret->region.begin());
        ret->dst_offset = dst_offset;

        return ret;
    }

    int playback(
        cl_command_queue queue) override
    {
        return g_pNextDispatch->clEnqueueCopyImageToBuffer(
            queue,
            src_image,
            dst_buffer,
            src_origin.data(),
            region.data(),
            dst_offset,
            0,
            NULL,
            0);
    }

    cl_mem src_image = nullptr;
    cl_mem dst_buffer = nullptr;
    std::array<size_t, 3> src_origin = {0, 0, 0};
    std::array<size_t, 3> region = {0, 0, 0};
    size_t dst_offset = 0;

private:
    CopyImageToBuffer() = default;
};

struct FillBuffer : Command
{
    static FillBuffer* create(
        cl_mem buffer,
        const void* pattern,
        size_t pattern_size,
        size_t offset,
        size_t size)
    {
        auto* ret = new FillBuffer();

        ret->buffer = buffer;

        auto* p = reinterpret_cast<const uint8_t*>(pattern);
        ret->pattern.reserve(pattern_size);
        ret->pattern.insert(
            ret->pattern.begin(),
            p,
            p + pattern_size);

        ret->offset = offset;
        ret->size = size;

        return ret;
    }

    int playback(
        cl_command_queue queue) override
    {
        return g_pNextDispatch->clEnqueueFillBuffer(
            queue,
            buffer,
            pattern.data(),
            pattern.size(),
            offset,
            size,
            0,
            NULL,
            0);
    }

    cl_mem buffer = nullptr;
    std::vector<uint8_t> pattern;
    size_t offset = 0;
    size_t size = 0;

private:
    FillBuffer() = default;
};

struct FillImage : Command
{
    static FillImage* create(
        cl_mem image,
        const void* fill_color,
        const size_t* origin,
        const size_t* region)
    {
        auto* ret = new FillImage();

        ret->image = image;

        auto* p = reinterpret_cast<const uint8_t*>(fill_color);

        size_t s = 0;
        g_pNextDispatch->clGetImageInfo(
            image,
            CL_IMAGE_ELEMENT_SIZE,
            sizeof(size_t),
            &s,
            nullptr);
        ret->fill_color.reserve(s);
        ret->fill_color.insert(
            ret->fill_color.begin(),
            p,
            p + s);

        std::copy(origin, origin + 3, ret->origin.begin());
        std::copy(region, region + 3, ret->region.begin());

        return ret;
    }

    int playback(
        cl_command_queue queue) override
    {
        return g_pNextDispatch->clEnqueueFillImage(
            queue,
            image,
            fill_color.data(),
            origin.data(),
            region.data(),
            0,
            NULL,
            0);
    }

    cl_mem image = nullptr;
    std::vector<uint8_t> fill_color;
    std::array<size_t, 3> origin = {0, 0, 0};
    std::array<size_t, 3> region = {0, 0, 0};

private:
    FillImage() = default;
};

struct NDRangeKernel : Command
{
    static NDRangeKernel* create(
        cl_kernel kernel,
        cl_uint work_dim,
        const size_t* global_work_offset,
        const size_t* global_work_size,
        const size_t* local_work_size)
    {
        auto* ret = new NDRangeKernel();

        ret->kernel = g_pNextDispatch->clCloneKernel(kernel, NULL);
        ret->work_dim = work_dim;

        if( global_work_offset )
        {
            ret->global_work_offset.reserve(work_dim);
            ret->global_work_offset.insert(
                ret->global_work_offset.begin(),
                global_work_offset,
                global_work_offset + work_dim);
        }

        if( global_work_size )
        {
            ret->global_work_size.reserve(work_dim);
            ret->global_work_size.insert(
                ret->global_work_size.begin(),
                global_work_size,
                global_work_size + work_dim);
        }

        if( local_work_size )
        {
            ret->local_work_size.reserve(work_dim);
            ret->local_work_size.insert(
                ret->local_work_size.begin(),
                local_work_size,
                local_work_size + work_dim);
        }

        return ret;
    }

    int playback(
        cl_command_queue queue) override
    {
        return g_pNextDispatch->clEnqueueNDRangeKernel(
            queue,
            kernel,
            work_dim,
            global_work_offset.size() ? global_work_offset.data() : NULL,
            global_work_size.data(),
            local_work_size.size() ? local_work_size.data() : NULL,
            0,
            NULL,
            0);
    }

    cl_kernel kernel = nullptr;
    cl_uint work_dim = 0;
    std::vector<size_t> global_work_offset;
    std::vector<size_t> global_work_size;
    std::vector<size_t> local_work_size;

private:
    NDRangeKernel() = default;
};

}; // namespace CmdBuf

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
        if( num_queues != 1 || queues == NULL )
        {
            errorCode = CL_INVALID_VALUE;
        }
        if( errcode_ret )
        {
            errcode_ret[0] = errorCode;
        }
        if( errorCode == CL_SUCCESS) {
            cmdbuf = new _cl_command_buffer_khr();
            cmdbuf->Queues.reserve(num_queues);
            cmdbuf->Queues.insert(
                cmdbuf->Queues.begin(),
                queues,
                queues + num_queues );
        }
        return cmdbuf;
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
                auto*   ptr = (cl_command_queue*)param_value;
                return writeVectorToMemory(
                    param_value_size,
                    Queues,
                    param_value_size_ret,
                    ptr );
            }
            break;
        case CL_COMMAND_BUFFER_NUM_QUEUES_KHR:
            {
                auto*   ptr = (cl_uint*)param_value;
                return writeParamToMemory(
                    param_value_size,
                    static_cast<cl_uint>(Queues.size()),
                    param_value_size_ret,
                    ptr );
            }
            break;
        case CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR:
            {
                auto*   ptr = (cl_uint*)param_value;
                return writeParamToMemory(
                    param_value_size,
                    RefCount.load(std::memory_order_relaxed),
                    param_value_size_ret,
                    ptr );
            }
            break;
        case CL_COMMAND_BUFFER_STATE_KHR:
            {
                auto*   ptr = (cl_command_buffer_state_khr*)param_value;
                return writeParamToMemory(
                    param_value_size,
                    State,
                    param_value_size_ret,
                    ptr );
            }
            break;
        case CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR:
            {
                auto*   ptr = (cl_command_buffer_properties_khr*)param_value;
                return writeVectorToMemory(
                    param_value_size,
                    {}, // No properties are currently supported.
                    param_value_size_ret,
                    ptr );
            }
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
        if( mutable_handle != NULL )
        {
            return CL_INVALID_VALUE;
        }
        if( ( sync_point_wait_list == NULL && num_sync_points_in_wait_list > 0 ) ||
            ( sync_point_wait_list != NULL && num_sync_points_in_wait_list == 0 ) )
        {
            return CL_INVALID_SYNC_POINT_WAIT_LIST_KHR;
        }

        uint32_t numSyncPoints = NextSyncPoint.load(std::memory_order_relaxed);
        for( cl_uint i = 0; i < num_sync_points_in_wait_list; i++ )
        {
            if( sync_point_wait_list[i] >= numSyncPoints )
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
                CmdBuf::Command* command,
                cl_sync_point_khr* sync_point )
    {
        Commands.push_back(command);

        if( sync_point != nullptr )
        {
            sync_point[0] = NextSyncPoint.fetch_add(1, std::memory_order_relaxed);
        }
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
        for( auto& command : Commands )
        {
            cl_int  errorCode = command->playback(queue);
            if( errorCode != CL_SUCCESS )
            {
                return errorCode;
            }
        }

        return CL_SUCCESS;
    }

private:
    static constexpr cl_uint cMagic = 0x434d4442;   // "CMDB"

    const cl_uint Magic;
    std::vector<cl_command_queue>   Queues;
    cl_command_buffer_state_khr State;
    std::atomic<uint32_t> RefCount;

    std::vector<CmdBuf::Command*> Commands;
    std::atomic<uint32_t> NextSyncPoint;

    _cl_command_buffer_khr() :
        Magic(cMagic),
        State(CL_COMMAND_BUFFER_STATE_RECORDING_KHR),
        RefCount(1),
        NextSyncPoint(0) {}
} CommandBuffer;

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

    if( errorCode == CL_SUCCESS && num_events_in_wait_list )
    {
        errorCode = g_pNextDispatch->clEnqueueBarrierWithWaitList(
            queue,
            num_events_in_wait_list,
            event_wait_list,
            NULL );
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
        CmdBuf::BarrierWithWaitList::create(),
        sync_point);
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
        CmdBuf::CopyBuffer::create(
            src_buffer,
            dst_buffer,
            src_offset,
            dst_offset,
            size),
        sync_point);
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
        CmdBuf::CopyBufferRect::create(
            src_buffer,
            dst_buffer,
            src_origin,
            dst_origin,
            region,
            src_row_pitch,
            src_slice_pitch,
            dst_row_pitch,
            dst_slice_pitch),
        sync_point);
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
        CmdBuf::CopyBufferToImage::create(
            src_buffer,
            dst_image,
            src_offset,
            dst_origin,
            region),
        sync_point);
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
        CmdBuf::CopyImage::create(
            src_image,
            dst_image,
            src_origin,
            dst_origin,
            region),
        sync_point);
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
        CmdBuf::CopyImageToBuffer::create(
            src_image,
            dst_buffer,
            src_origin,
            region,
            dst_offset),
        sync_point);
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
        CmdBuf::FillBuffer::create(
            buffer,
            pattern,
            pattern_size,
            offset,
            size),
        sync_point);
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
        CmdBuf::FillImage::create(
            image,
            fill_color,
            origin,
            region),
        sync_point);
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

    cmdbuf->addCommand(
        CmdBuf::NDRangeKernel::create(
            kernel,
            work_dim,
            global_work_offset,
            global_work_size,
            local_work_size),
        sync_point);
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
