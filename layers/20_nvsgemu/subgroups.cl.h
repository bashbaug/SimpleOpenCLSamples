const char* g_NVSubGroupString = R"CLC(

#define cl_khr_subgroups
#define __opencl_c_subgroups
#if defined(cl_khr_subgroups)
// Basic cl_khr_subgroup functions:

uint    __attribute__((overloadable)) get_max_sub_group_size( void )
{
    uint ret;
    asm ("mov.u32 %0, WARP_SZ;" : "=r"(ret));
    return ret;
}

uint    __attribute__((overloadable)) get_num_sub_groups( void )
{
    uint lws = get_local_size(0);
    lws *= (uint)get_local_size(1);
    lws *= (uint)get_local_size(2);
    return (lws + get_max_sub_group_size() - 1) / get_max_sub_group_size();
}

uint    __attribute__((overloadable)) get_sub_group_id( void )
{
    uint llid = get_local_linear_id();
    return llid / get_max_sub_group_size();
}

uint    __attribute__((overloadable)) get_sub_group_local_id( void )
{
    uint ret;
    asm ("mov.u32 %0, %%laneid;" : "=r"(ret));
    return ret;
}

uint    __attribute__((overloadable)) get_sub_group_size( void )
{
    uint ret = get_max_sub_group_size();
    if (get_sub_group_id() == get_num_sub_groups() - 1)
    {
        uint lws = get_local_size(0);
        lws *= (uint)get_local_size(1);
        lws *= (uint)get_local_size(2);
        ret = lws % get_max_sub_group_size();
    }
    return ret;
}

void    __attribute__((overloadable)) sub_group_barrier( cl_mem_fence_flags flags )
{
    asm volatile("bar.warp.sync 0xFFFFFFFF;" : : : "memory");
}

#if (__OPENCL_C_VERSION__ >= CL_VERSION_2_0)
uint    __attribute__((overloadable)) get_enqueued_num_sub_groups( void )
{
    uint lws = get_enqueued_local_size(0);
    lws *= (uint)get_enqueued_local_size(1);
    lws *= (uint)get_enqueued_local_size(2);
    return (lws + get_max_sub_group_size() - 1) / get_max_sub_group_size();
}

void    __attribute__((overloadable)) sub_group_barrier( cl_mem_fence_flags flags, memory_scope scope )
{
    asm volatile("bar.warp.sync 0xFFFFFFFF;" : : : "memory");
}
#endif

int     __attribute__((overloadable)) sub_group_all( int predicate )
{
    int ret;
    asm ("{ .reg .pred p, q; \
        setp.ne.u32 p, %1, 0; \
        vote.sync.all.pred q, p, 0xFFFFFFFF; \
        selp.u32 %0, 1, 0, q; }" : "=r"(ret) : "r"(predicate));
    return ret;
}
int     __attribute__((overloadable)) sub_group_any( int predicate )
{
    int ret;
    asm ("{ .reg .pred p, q; \
        setp.ne.u32 p, %1, 0; \
        vote.sync.any.pred q, p, 0xFFFFFFFF; \
        selp.u32 %0, 1, 0, q; }" : "=r"(ret) : "r"(predicate));
    return ret;
}

// Broadcast:

int     __attribute__((overloadable)) sub_group_broadcast( int   x, uint index )
{
    uint ret;
    asm ("{ .reg .u32 t; \
        mov.b32 t, %1; \
        shfl.sync.idx.b32 %0, t, %2, 31, 0xFFFFFFFF; }" : "=r"(ret) : "r"(x), "r"(index));
    return ret;
}

uint    __attribute__((overloadable)) sub_group_broadcast( uint  x, uint index )
{
    uint ret;
    asm ("{ .reg .u32 t; \
        mov.b32 t, %1; \
        shfl.sync.idx.b32 %0, t, %2, 31, 0xFFFFFFFF; }" : "=r"(ret) : "r"(x), "r"(index));
    return ret;
}

long    __attribute__((overloadable)) sub_group_broadcast( long  x, uint index )
{
    long ret;
    asm ("{ .reg .u32 th, tl; \
        mov.b64 {th, tl}, %1; \
        shfl.sync.idx.b32 th, th, %2, 31, 0xFFFFFFFF; \
        shfl.sync.idx.b32 tl, tl, %2, 31, 0xFFFFFFFF; \
        mov.b64 %0, {th, tl}; }" : "=l"(ret) : "l"(x), "r"(index));
    return ret;
}

ulong   __attribute__((overloadable)) sub_group_broadcast( ulong x, uint index )
{
    ulong ret;
    asm ("{ .reg .u32 th, tl; \
        mov.b64 {th, tl}, %1; \
        shfl.sync.idx.b32 th, th, %2, 31, 0xFFFFFFFF; \
        shfl.sync.idx.b32 tl, tl, %2, 31, 0xFFFFFFFF; \
        mov.b64 %0, {th, tl}; }" : "=l"(ret) : "l"(x), "r"(index));
    return ret;
}

float   __attribute__((overloadable)) sub_group_broadcast( float x, uint index )
{
    float ret;
    asm ("{ .reg .f32 t; \
        mov.b32 t, %1; shfl.sync.idx.b32 %0, t, %2, 31, 0xFFFFFFFF; }" : "=f"(ret) : "f"(x), "r"(index));
    return ret;
}

#ifdef cl_khr_fp16
half __attribute__((overloadable)) sub_group_broadcast(half x, uint index);
#endif
#if defined(cl_khr_fp64)
double __attribute__((overloadable)) sub_group_broadcast(double x, uint index)
{
    double ret;
    asm ("{ .reg .u32 th, tl; \
        mov.b64 {th, tl}, %1; \
        shfl.sync.idx.b32 th, th, %2, 31, 0xFFFFFFFF; \
        shfl.sync.idx.b32 tl, tl, %2, 31, 0xFFFFFFFF; \
        mov.b64 %0, {th, tl}; }" : "=l"(ret) : "l"(x), "r"(index));
    return ret;
}
#endif

// Scans and reductions:

#define SG_FUNC_HELPER(_type, _iadd, _imin, _imax)                      \
_type __attribute__((overloadable)) sub_group_reduce_add(_type x) {     \
    _type ret = sub_group_broadcast(x, 0);                              \
    for (uint i = 1; i < get_sub_group_size(); i++) {                   \
        ret += sub_group_broadcast(x, i);                               \
    }                                                                   \
    return ret;                                                         \
}                                                                       \
_type __attribute__((overloadable)) sub_group_reduce_min(_type x) {     \
    _type ret = sub_group_broadcast(x, 0);                              \
    for (uint i = 1; i < get_sub_group_size(); i++) {                   \
        ret = min(ret, sub_group_broadcast(x, i));                      \
    }                                                                   \
    return ret;                                                         \
}                                                                       \
_type __attribute__((overloadable)) sub_group_reduce_max(_type x) {     \
    _type ret = sub_group_broadcast(x, 0);                              \
    for (uint i = 1; i < get_sub_group_size(); i++) {                   \
        ret = max(ret, sub_group_broadcast(x, i));                      \
    }                                                                   \
    return ret;                                                         \
}                                                                       \
_type __attribute__((overloadable)) sub_group_scan_exclusive_add(_type x) { \
    _type ret = _iadd;                                                  \
    for (uint i = 0; i < get_max_sub_group_size(); i++) {               \
        _type val = sub_group_broadcast(x, i);                          \
        val = i < get_sub_group_local_id() ? val : _iadd;               \
        ret += val;                                                     \
    }                                                                   \
    return ret;                                                         \
}                                                                       \
_type __attribute__((overloadable)) sub_group_scan_exclusive_min(_type x) { \
    _type ret = _imin;                                                  \
    for (uint i = 0; i < get_max_sub_group_size(); i++) {               \
        _type val = sub_group_broadcast(x, i);                          \
        val = i < get_sub_group_local_id() ? val : _imin;               \
        ret = min(ret, val);                                            \
    }                                                                   \
    return ret;                                                         \
}                                                                       \
_type __attribute__((overloadable)) sub_group_scan_exclusive_max(_type x) { \
    _type ret = _imax;                                                  \
    for (uint i = 0; i < get_max_sub_group_size(); i++) {               \
        _type val = sub_group_broadcast(x, i);                          \
        val = i < get_sub_group_local_id() ? val : _imax;               \
        ret = max(ret, val);                                            \
    }                                                                   \
    return ret;                                                         \
}                                                                       \
_type __attribute__((overloadable)) sub_group_scan_inclusive_add(_type x) { \
    _type ret = _iadd;                                                  \
    for (uint i = 0; i < get_max_sub_group_size(); i++) {               \
        _type val = sub_group_broadcast(x, i);                          \
        val = i <= get_sub_group_local_id() ? val : _iadd;              \
        ret += val;                                                     \
    }                                                                   \
    return ret;                                                         \
}                                                                       \
_type __attribute__((overloadable)) sub_group_scan_inclusive_min(_type x) { \
    _type ret = _imin;                                                  \
    for (uint i = 0; i < get_max_sub_group_size(); i++) {               \
        _type val = sub_group_broadcast(x, i);                          \
        val = i <= get_sub_group_local_id() ? val : _imin;              \
        ret = min(ret, val);                                            \
    }                                                                   \
    return ret;                                                         \
}                                                                       \
_type __attribute__((overloadable)) sub_group_scan_inclusive_max(_type x) { \
    _type ret = _imax;                                                  \
    for (uint i = 0; i < get_max_sub_group_size(); i++) {               \
        _type val = sub_group_broadcast(x, i);                          \
        val = i <= get_sub_group_local_id() ? val : _imax;              \
        ret = max(ret, val);                                            \
    }                                                                   \
    return ret;                                                         \
}                                                                       \

SG_FUNC_HELPER(int, 0, INT_MAX, INT_MIN)
SG_FUNC_HELPER(uint, 0, UINT_MAX, 0)
SG_FUNC_HELPER(long, 0, LONG_MAX, LONG_MIN)
SG_FUNC_HELPER(ulong, 0, ULONG_MAX, 0)
SG_FUNC_HELPER(float, 0.0f, INFINITY, -INFINITY)

#ifdef cl_khr_fp16
SG_FUNC_HELPER(half, 0.0h, INFINITY, -INFINITY)
#endif

#if defined(cl_khr_fp64)
SG_FUNC_HELPER(double, 0.0, INFINITY, -INFINITY)
#endif

#endif // defined(cl_khr_subgroups)

// Extended subgroups:

#if defined(cl_khr_subgroup_shuffle)
    // Shuffle:
    char    __attribute__((overloadable)) sub_group_shuffle(char value, uint index);
    uchar   __attribute__((overloadable)) sub_group_shuffle(uchar value, uint index);
    int     __attribute__((overloadable)) sub_group_shuffle(int value, uint index);
    uint    __attribute__((overloadable)) sub_group_shuffle(uint value, uint index);
    long    __attribute__((overloadable)) sub_group_shuffle(long value, uint index);
    ulong   __attribute__((overloadable)) sub_group_shuffle(ulong value, uint index);
    float   __attribute__((overloadable)) sub_group_shuffle(float value, uint index);
    #if defined(cl_khr_fp64)
    double  __attribute__((overloadable)) sub_group_shuffle(double value, uint index);
    #endif // defined(cl_khr_fp64)
    #if defined (cl_khr_fp16)
    half    __attribute__((overloadable)) sub_group_shuffle(half value, uint index);
    #endif // defined (cl_khr_fp16)

    // Shuffle XOR:
    char    __attribute__((overloadable)) sub_group_shuffle_xor(char value, uint mask);
    uchar   __attribute__((overloadable)) sub_group_shuffle_xor(uchar value, uint mask);
    int     __attribute__((overloadable)) sub_group_shuffle_xor(int value, uint mask);
    uint    __attribute__((overloadable)) sub_group_shuffle_xor(uint value, uint mask);
    long    __attribute__((overloadable)) sub_group_shuffle_xor(long value, uint mask);
    ulong   __attribute__((overloadable)) sub_group_shuffle_xor(ulong value, uint mask);
    float   __attribute__((overloadable)) sub_group_shuffle_xor(float value, uint mask);
    #if defined(cl_khr_fp64)
    double  __attribute__((overloadable)) sub_group_shuffle_xor(double value, uint mask);
    #endif // defined(cl_khr_fp64)
    #if defined (cl_khr_fp16)
    half    __attribute__((overloadable)) sub_group_shuffle_xor(half value, uint mask);
    #endif // defined (cl_khr_fp16)
#endif // defined(cl_khr_subgroup_shuffle)

#if defined(cl_khr_subgroup_shuffle_relative)
    // Shuffle up:
    char    __attribute__((overloadable)) sub_group_shuffle_up(char value, uint delta);
    uchar   __attribute__((overloadable)) sub_group_shuffle_up(uchar value, uint delta);
    int     __attribute__((overloadable)) sub_group_shuffle_up(int value, uint delta);
    uint    __attribute__((overloadable)) sub_group_shuffle_up(uint value, uint delta);
    long    __attribute__((overloadable)) sub_group_shuffle_up(long value, uint delta);
    ulong   __attribute__((overloadable)) sub_group_shuffle_up(ulong value, uint delta);
    float   __attribute__((overloadable)) sub_group_shuffle_up(float value, uint delta);
    #if defined(cl_khr_fp64)
    double  __attribute__((overloadable)) sub_group_shuffle_up(double value, uint delta);
    #endif // defined(cl_khr_fp64)
    #if defined(cl_khr_fp16)
    half    __attribute__((overloadable)) sub_group_shuffle_up(half value, uint delta);
    #endif // defined (cl_khr_fp16)

    // Shuffle down:
    char    __attribute__((overloadable)) sub_group_shuffle_down(char value, uint delta);
    uchar   __attribute__((overloadable)) sub_group_shuffle_down(uchar value, uint delta);
    int     __attribute__((overloadable)) sub_group_shuffle_down(int value, uint delta);
    uint    __attribute__((overloadable)) sub_group_shuffle_down(uint value, uint delta);
    long    __attribute__((overloadable)) sub_group_shuffle_down(long value, uint delta);
    ulong   __attribute__((overloadable)) sub_group_shuffle_down(ulong value, uint delta);
    float   __attribute__((overloadable)) sub_group_shuffle_down(float value, uint delta);
    #if defined(cl_khr_fp64)
    double  __attribute__((overloadable)) sub_group_shuffle_down(double value, uint delta);
    #endif // defined(cl_khr_fp64)
    #if defined(cl_khr_fp16)
    half    __attribute__((overloadable)) sub_group_shuffle_down(half value, uint delta);
    #endif // defined (cl_khr_fp16)
#endif // defined(cl_khr_subgroup_shuffle_relative)

#if defined(cl_khr_subgroup_non_uniform_vote)
    int     __attribute__((overloadable)) sub_group_elect(void);
    int     __attribute__((overloadable)) sub_group_non_uniform_all(int predicate);
    int     __attribute__((overloadable)) sub_group_non_uniform_any(int predicate);
    int     __attribute__((overloadable)) sub_group_non_uniform_all_equal(char value);
    int     __attribute__((overloadable)) sub_group_non_uniform_all_equal(uchar value);
    int     __attribute__((overloadable)) sub_group_non_uniform_all_equal(short value);
    int     __attribute__((overloadable)) sub_group_non_uniform_all_equal(ushort value);
    int     __attribute__((overloadable)) sub_group_non_uniform_all_equal(int value);
    int     __attribute__((overloadable)) sub_group_non_uniform_all_equal(uint value);
    int     __attribute__((overloadable)) sub_group_non_uniform_all_equal(long value);
    int     __attribute__((overloadable)) sub_group_non_uniform_all_equal(ulong value);
    int     __attribute__((overloadable)) sub_group_non_uniform_all_equal(float value);
    #if defined(cl_khr_fp64)
    int     __attribute__((overloadable)) sub_group_non_uniform_all_equal(double value);
    #endif // defined(cl_khr_fp64)
    #if defined(cl_khr_fp16)
    int     __attribute__((overloadable)) sub_group_non_uniform_all_equal(half value);
    #endif // defined(cl_khr_fp16)
#endif // defined(cl_khr_subgroup_non_uniform_vote)

#if defined(cl_khr_subgroup_ballot)
    char __attribute__((overloadable)) sub_group_non_uniform_broadcast(char value, uint index);
    char __attribute__((overloadable)) sub_group_broadcast_first(char value);
    char2 __attribute__((overloadable)) sub_group_non_uniform_broadcast(char2 value, uint index);
    char2 __attribute__((overloadable)) sub_group_broadcast_first(char2 value);
    char3 __attribute__((overloadable)) sub_group_non_uniform_broadcast(char3 value, uint index);
    char3 __attribute__((overloadable)) sub_group_broadcast_first(char3 value);
    char4 __attribute__((overloadable)) sub_group_non_uniform_broadcast(char4 value, uint index);
    char4 __attribute__((overloadable)) sub_group_broadcast_first(char4 value);
    char8 __attribute__((overloadable)) sub_group_non_uniform_broadcast(char8 value, uint index);
    char8 __attribute__((overloadable)) sub_group_broadcast_first(char8 value);
    char16 __attribute__((overloadable)) sub_group_non_uniform_broadcast(char16 value, uint index);
    char16 __attribute__((overloadable)) sub_group_broadcast_first(char16 value);

    uchar __attribute__((overloadable)) sub_group_non_uniform_broadcast(uchar value, uint index);
    uchar __attribute__((overloadable)) sub_group_broadcast_first(uchar value);
    uchar2 __attribute__((overloadable)) sub_group_non_uniform_broadcast(uchar2 value, uint index);
    uchar2 __attribute__((overloadable)) sub_group_broadcast_first(uchar2 value);
    uchar3 __attribute__((overloadable)) sub_group_non_uniform_broadcast(uchar3 value, uint index);
    uchar3 __attribute__((overloadable)) sub_group_broadcast_first(uchar3 value);
    uchar4 __attribute__((overloadable)) sub_group_non_uniform_broadcast(uchar4 value, uint index);
    uchar4 __attribute__((overloadable)) sub_group_broadcast_first(uchar4 value);
    uchar8 __attribute__((overloadable)) sub_group_non_uniform_broadcast(uchar8 value, uint index);
    uchar8 __attribute__((overloadable)) sub_group_broadcast_first(uchar8 value);
    uchar16 __attribute__((overloadable)) sub_group_non_uniform_broadcast(uchar16 value, uint index);
    uchar16 __attribute__((overloadable)) sub_group_broadcast_first(uchar16 value);

    short __attribute__((overloadable)) sub_group_non_uniform_broadcast(short value, uint index);
    short __attribute__((overloadable)) sub_group_broadcast_first(short value);
    short2 __attribute__((overloadable)) sub_group_non_uniform_broadcast(short2 value, uint index);
    short2 __attribute__((overloadable)) sub_group_broadcast_first(short2 value);
    short3 __attribute__((overloadable)) sub_group_non_uniform_broadcast(short3 value, uint index);
    short3 __attribute__((overloadable)) sub_group_broadcast_first(short3 value);
    short4 __attribute__((overloadable)) sub_group_non_uniform_broadcast(short4 value, uint index);
    short4 __attribute__((overloadable)) sub_group_broadcast_first(short4 value);
    short8 __attribute__((overloadable)) sub_group_non_uniform_broadcast(short8 value, uint index);
    short8 __attribute__((overloadable)) sub_group_broadcast_first(short8 value);
    short16 __attribute__((overloadable)) sub_group_non_uniform_broadcast(short16 value, uint index);
    short16 __attribute__((overloadable)) sub_group_broadcast_first(short16 value);

    ushort __attribute__((overloadable)) sub_group_non_uniform_broadcast(ushort value, uint index);
    ushort __attribute__((overloadable)) sub_group_broadcast_first(ushort value);
    ushort2 __attribute__((overloadable)) sub_group_non_uniform_broadcast(ushort2 value, uint index);
    ushort2 __attribute__((overloadable)) sub_group_broadcast_first(ushort2 value);
    ushort3 __attribute__((overloadable)) sub_group_non_uniform_broadcast(ushort3 value, uint index);
    ushort3 __attribute__((overloadable)) sub_group_broadcast_first(ushort3 value);
    ushort4 __attribute__((overloadable)) sub_group_non_uniform_broadcast(ushort4 value, uint index);
    ushort4 __attribute__((overloadable)) sub_group_broadcast_first(ushort4 value);
    ushort8 __attribute__((overloadable)) sub_group_non_uniform_broadcast(ushort8 value, uint index);
    ushort8 __attribute__((overloadable)) sub_group_broadcast_first(ushort8 value);
    ushort16 __attribute__((overloadable)) sub_group_non_uniform_broadcast(ushort16 value, uint index);
    ushort16 __attribute__((overloadable)) sub_group_broadcast_first(ushort16 value);

    int __attribute__((overloadable)) sub_group_non_uniform_broadcast(int value, uint index);
    int __attribute__((overloadable)) sub_group_broadcast_first(int value);
    int2 __attribute__((overloadable)) sub_group_non_uniform_broadcast(int2 value, uint index);
    int2 __attribute__((overloadable)) sub_group_broadcast_first(int2 value);
    int3 __attribute__((overloadable)) sub_group_non_uniform_broadcast(int3 value, uint index);
    int3 __attribute__((overloadable)) sub_group_broadcast_first(int3 value);
    int4 __attribute__((overloadable)) sub_group_non_uniform_broadcast(int4 value, uint index);
    int4 __attribute__((overloadable)) sub_group_broadcast_first(int4 value);
    int8 __attribute__((overloadable)) sub_group_non_uniform_broadcast(int8 value, uint index);
    int8 __attribute__((overloadable)) sub_group_broadcast_first(int8 value);
    int16 __attribute__((overloadable)) sub_group_non_uniform_broadcast(int16 value, uint index);
    int16 __attribute__((overloadable)) sub_group_broadcast_first(int16 value);

    uint __attribute__((overloadable)) sub_group_non_uniform_broadcast(uint value, uint index);
    uint __attribute__((overloadable)) sub_group_broadcast_first(uint value);
    uint2 __attribute__((overloadable)) sub_group_non_uniform_broadcast(uint2 value, uint index);
    uint2 __attribute__((overloadable)) sub_group_broadcast_first(uint2 value);
    uint3 __attribute__((overloadable)) sub_group_non_uniform_broadcast(uint3 value, uint index);
    uint3 __attribute__((overloadable)) sub_group_broadcast_first(uint3 value);
    uint4 __attribute__((overloadable)) sub_group_non_uniform_broadcast(uint4 value, uint index);
    uint4 __attribute__((overloadable)) sub_group_broadcast_first(uint4 value);
    uint8 __attribute__((overloadable)) sub_group_non_uniform_broadcast(uint8 value, uint index);
    uint8 __attribute__((overloadable)) sub_group_broadcast_first(uint8 value);
    uint16 __attribute__((overloadable)) sub_group_non_uniform_broadcast(uint16 value, uint index);
    uint16 __attribute__((overloadable)) sub_group_broadcast_first(uint16 value);

    long __attribute__((overloadable)) sub_group_non_uniform_broadcast(long value, uint index);
    long __attribute__((overloadable)) sub_group_broadcast_first(long value);
    long2 __attribute__((overloadable)) sub_group_non_uniform_broadcast(long2 value, uint index);
    long2 __attribute__((overloadable)) sub_group_broadcast_first(long2 value);
    long3 __attribute__((overloadable)) sub_group_non_uniform_broadcast(long3 value, uint index);
    long3 __attribute__((overloadable)) sub_group_broadcast_first(long3 value);
    long4 __attribute__((overloadable)) sub_group_non_uniform_broadcast(long4 value, uint index);
    long4 __attribute__((overloadable)) sub_group_broadcast_first(long4 value);
    long8 __attribute__((overloadable)) sub_group_non_uniform_broadcast(long8 value, uint index);
    long8 __attribute__((overloadable)) sub_group_broadcast_first(long8 value);
    long16 __attribute__((overloadable)) sub_group_non_uniform_broadcast(long16 value, uint index);
    long16 __attribute__((overloadable)) sub_group_broadcast_first(long16 value);

    ulong __attribute__((overloadable)) sub_group_non_uniform_broadcast(ulong value, uint index);
    ulong __attribute__((overloadable)) sub_group_broadcast_first(ulong value);
    ulong2 __attribute__((overloadable)) sub_group_non_uniform_broadcast(ulong2 value, uint index);
    ulong2 __attribute__((overloadable)) sub_group_broadcast_first(ulong2 value);
    ulong3 __attribute__((overloadable)) sub_group_non_uniform_broadcast(ulong3 value, uint index);
    ulong3 __attribute__((overloadable)) sub_group_broadcast_first(ulong3 value);
    ulong4 __attribute__((overloadable)) sub_group_non_uniform_broadcast(ulong4 value, uint index);
    ulong4 __attribute__((overloadable)) sub_group_broadcast_first(ulong4 value);
    ulong8 __attribute__((overloadable)) sub_group_non_uniform_broadcast(ulong8 value, uint index);
    ulong8 __attribute__((overloadable)) sub_group_broadcast_first(ulong8 value);
    ulong16 __attribute__((overloadable)) sub_group_non_uniform_broadcast(ulong16 value, uint index);
    ulong16 __attribute__((overloadable)) sub_group_broadcast_first(ulong16 value);

    float __attribute__((overloadable)) sub_group_non_uniform_broadcast(float value, uint index);
    float __attribute__((overloadable)) sub_group_broadcast_first(float value);
    float2 __attribute__((overloadable)) sub_group_non_uniform_broadcast(float2 value, uint index);
    float2 __attribute__((overloadable)) sub_group_broadcast_first(float2 value);
    float3 __attribute__((overloadable)) sub_group_non_uniform_broadcast(float3 value, uint index);
    float3 __attribute__((overloadable)) sub_group_broadcast_first(float3 value);
    float4 __attribute__((overloadable)) sub_group_non_uniform_broadcast(float4 value, uint index);
    float4 __attribute__((overloadable)) sub_group_broadcast_first(float4 value);
    float8 __attribute__((overloadable)) sub_group_non_uniform_broadcast(float8 value, uint index);
    float8 __attribute__((overloadable)) sub_group_broadcast_first(float8 value);
    float16 __attribute__((overloadable)) sub_group_non_uniform_broadcast(float16 value, uint index);
    float16 __attribute__((overloadable)) sub_group_broadcast_first(float16 value);

    #if defined(cl_khr_fp64)
    double __attribute__((overloadable)) sub_group_non_uniform_broadcast(double value, uint index);
    double __attribute__((overloadable)) sub_group_broadcast_first(double value);
    double2 __attribute__((overloadable)) sub_group_non_uniform_broadcast(double2 value, uint index);
    double2 __attribute__((overloadable)) sub_group_broadcast_first(double2 value);
    double3 __attribute__((overloadable)) sub_group_non_uniform_broadcast(double3 value, uint index);
    double3 __attribute__((overloadable)) sub_group_broadcast_first(double3 value);
    double4 __attribute__((overloadable)) sub_group_non_uniform_broadcast(double4 value, uint index);
    double4 __attribute__((overloadable)) sub_group_broadcast_first(double4 value);
    double8 __attribute__((overloadable)) sub_group_non_uniform_broadcast(double8 value, uint index);
    double8 __attribute__((overloadable)) sub_group_broadcast_first(double8 value);
    double16 __attribute__((overloadable)) sub_group_non_uniform_broadcast(double16 value, uint index);
    double16 __attribute__((overloadable)) sub_group_broadcast_first(double16 value);
    #endif // defined(cl_khr_fp64)

    #if defined(cl_khr_fp16)
    half __attribute__((overloadable)) sub_group_non_uniform_broadcast(half value, uint index);
    half __attribute__((overloadable)) sub_group_broadcast_first(half value);
    half2 __attribute__((overloadable)) sub_group_non_uniform_broadcast(half2 value, uint index);
    half2 __attribute__((overloadable)) sub_group_broadcast_first(half2 value);
    half3 __attribute__((overloadable)) sub_group_non_uniform_broadcast(half3 value, uint index);
    half3 __attribute__((overloadable)) sub_group_broadcast_first(half3 value);
    half4 __attribute__((overloadable)) sub_group_non_uniform_broadcast(half4 value, uint index);
    half4 __attribute__((overloadable)) sub_group_broadcast_first(half4 value);
    half8 __attribute__((overloadable)) sub_group_non_uniform_broadcast(half8 value, uint index);
    half8 __attribute__((overloadable)) sub_group_broadcast_first(half8 value);
    half16 __attribute__((overloadable)) sub_group_non_uniform_broadcast(half16 value, uint index);
    half16 __attribute__((overloadable)) sub_group_broadcast_first(half16 value);
    #endif // defined(cl_khr_fp16)

    uint4 __attribute__((overloadable)) sub_group_ballot(int predicate);
    int __attribute__((overloadable)) sub_group_inverse_ballot(uint4 value);
    int __attribute__((overloadable)) sub_group_ballot_bit_extract(uint4 value, uint index);
    uint __attribute__((overloadable)) sub_group_ballot_bit_count(uint4 value);
    uint __attribute__((overloadable)) sub_group_ballot_inclusive_scan(uint4 value);
    uint __attribute__((overloadable)) sub_group_ballot_exclusive_scan(uint4 value);
    uint __attribute__((overloadable)) sub_group_ballot_find_lsb(uint4 value);
    uint __attribute__((overloadable)) sub_group_ballot_find_msb(uint4 value);
    uint4 __attribute__((overloadable)) get_sub_group_eq_mask(void);
    uint4 __attribute__((overloadable)) get_sub_group_ge_mask(void);
    uint4 __attribute__((overloadable)) get_sub_group_gt_mask(void);
    uint4 __attribute__((overloadable)) get_sub_group_le_mask(void);
    uint4 __attribute__((overloadable)) get_sub_group_lt_mask(void);
#endif // defined(cl_khr_subgroup_ballot)

#if defined(cl_khr_subgroup_extended_types)
    char    __attribute__((overloadable)) sub_group_broadcast( char   x, uint index );
    uchar   __attribute__((overloadable)) sub_group_broadcast( uchar  x, uint index );
    short   __attribute__((overloadable)) sub_group_broadcast( short  x, uint index );
    ushort  __attribute__((overloadable)) sub_group_broadcast( ushort x, uint index );

    char2  __attribute__((overloadable)) sub_group_broadcast( char2  x, uint index );
    char3  __attribute__((overloadable)) sub_group_broadcast( char3  x, uint index );
    char4  __attribute__((overloadable)) sub_group_broadcast( char4  x, uint index );
    char8  __attribute__((overloadable)) sub_group_broadcast( char8  x, uint index );
    char16 __attribute__((overloadable)) sub_group_broadcast( char16 x, uint index );

    uchar2  __attribute__((overloadable)) sub_group_broadcast( uchar2  x, uint index );
    uchar3  __attribute__((overloadable)) sub_group_broadcast( uchar3  x, uint index );
    uchar4  __attribute__((overloadable)) sub_group_broadcast( uchar4  x, uint index );
    uchar8  __attribute__((overloadable)) sub_group_broadcast( uchar8  x, uint index );
    uchar16 __attribute__((overloadable)) sub_group_broadcast( uchar16 x, uint index );

    short2  __attribute__((overloadable)) sub_group_broadcast( short2  x, uint index );
    short3  __attribute__((overloadable)) sub_group_broadcast( short3  x, uint index );
    short4  __attribute__((overloadable)) sub_group_broadcast( short4  x, uint index );
    short8  __attribute__((overloadable)) sub_group_broadcast( short8  x, uint index );
    short16 __attribute__((overloadable)) sub_group_broadcast( short16 x, uint index );

    ushort2  __attribute__((overloadable)) sub_group_broadcast( ushort2  x, uint index );
    ushort3  __attribute__((overloadable)) sub_group_broadcast( ushort3  x, uint index );
    ushort4  __attribute__((overloadable)) sub_group_broadcast( ushort4  x, uint index );
    ushort8  __attribute__((overloadable)) sub_group_broadcast( ushort8  x, uint index );
    ushort16 __attribute__((overloadable)) sub_group_broadcast( ushort16 x, uint index );

    int2  __attribute__((overloadable)) sub_group_broadcast( int2  x, uint index );
    int3  __attribute__((overloadable)) sub_group_broadcast( int3  x, uint index );
    int4  __attribute__((overloadable)) sub_group_broadcast( int4  x, uint index );
    int8  __attribute__((overloadable)) sub_group_broadcast( int8  x, uint index );
    int16 __attribute__((overloadable)) sub_group_broadcast( int16 x, uint index );

    uint2  __attribute__((overloadable)) sub_group_broadcast( uint2  x, uint index );
    uint3  __attribute__((overloadable)) sub_group_broadcast( uint3  x, uint index );
    uint4  __attribute__((overloadable)) sub_group_broadcast( uint4  x, uint index );
    uint8  __attribute__((overloadable)) sub_group_broadcast( uint8  x, uint index );
    uint16 __attribute__((overloadable)) sub_group_broadcast( uint16 x, uint index );

    long2  __attribute__((overloadable)) sub_group_broadcast( long2  x, uint index );
    long3  __attribute__((overloadable)) sub_group_broadcast( long3  x, uint index );
    long4  __attribute__((overloadable)) sub_group_broadcast( long4  x, uint index );
    long8  __attribute__((overloadable)) sub_group_broadcast( long8  x, uint index );
    long16 __attribute__((overloadable)) sub_group_broadcast( long16 x, uint index );

    ulong2  __attribute__((overloadable)) sub_group_broadcast( ulong2  x, uint index );
    ulong3  __attribute__((overloadable)) sub_group_broadcast( ulong3  x, uint index );
    ulong4  __attribute__((overloadable)) sub_group_broadcast( ulong4  x, uint index );
    ulong8  __attribute__((overloadable)) sub_group_broadcast( ulong8  x, uint index );
    ulong16 __attribute__((overloadable)) sub_group_broadcast( ulong16 x, uint index );

    float2  __attribute__((overloadable)) sub_group_broadcast( float2  x, uint index );
    float3  __attribute__((overloadable)) sub_group_broadcast( float3  x, uint index );
    float4  __attribute__((overloadable)) sub_group_broadcast( float4  x, uint index );
    float8  __attribute__((overloadable)) sub_group_broadcast( float8  x, uint index );
    float16 __attribute__((overloadable)) sub_group_broadcast( float16 x, uint index );

    #ifdef cl_khr_fp16
    half2  __attribute__((overloadable)) sub_group_broadcast( half2  x, uint index );
    half3  __attribute__((overloadable)) sub_group_broadcast( half3  x, uint index );
    half4  __attribute__((overloadable)) sub_group_broadcast( half4  x, uint index );
    half8  __attribute__((overloadable)) sub_group_broadcast( half8  x, uint index );
    half16 __attribute__((overloadable)) sub_group_broadcast( half16 x, uint index );
    #endif

    #if defined(cl_khr_fp64)
    double2  __attribute__((overloadable)) sub_group_broadcast( double2  x, uint index );
    double3  __attribute__((overloadable)) sub_group_broadcast( double3  x, uint index );
    double4  __attribute__((overloadable)) sub_group_broadcast( double4  x, uint index );
    double8  __attribute__((overloadable)) sub_group_broadcast( double8  x, uint index );
    double16 __attribute__((overloadable)) sub_group_broadcast( double16 x, uint index );
    #endif

    char    __attribute__((overloadable)) sub_group_reduce_add(char x);
    char    __attribute__((overloadable)) sub_group_reduce_min(char x);
    char    __attribute__((overloadable)) sub_group_reduce_max(char x);
    char    __attribute__((overloadable)) sub_group_scan_exclusive_add(char x);
    char    __attribute__((overloadable)) sub_group_scan_exclusive_min(char x);
    char    __attribute__((overloadable)) sub_group_scan_exclusive_max(char x);
    char    __attribute__((overloadable)) sub_group_scan_inclusive_add(char x);
    char    __attribute__((overloadable)) sub_group_scan_inclusive_min(char x);
    char    __attribute__((overloadable)) sub_group_scan_inclusive_max(char x);

    uchar    __attribute__((overloadable)) sub_group_reduce_add(uchar x);
    uchar    __attribute__((overloadable)) sub_group_reduce_min(uchar x);
    uchar    __attribute__((overloadable)) sub_group_reduce_max(uchar x);
    uchar    __attribute__((overloadable)) sub_group_scan_exclusive_add(uchar x);
    uchar    __attribute__((overloadable)) sub_group_scan_exclusive_min(uchar x);
    uchar    __attribute__((overloadable)) sub_group_scan_exclusive_max(uchar x);
    uchar    __attribute__((overloadable)) sub_group_scan_inclusive_add(uchar x);
    uchar    __attribute__((overloadable)) sub_group_scan_inclusive_min(uchar x);
    uchar    __attribute__((overloadable)) sub_group_scan_inclusive_max(uchar x);

    short    __attribute__((overloadable)) sub_group_reduce_add(short x);
    short    __attribute__((overloadable)) sub_group_reduce_min(short x);
    short    __attribute__((overloadable)) sub_group_reduce_max(short x);
    short    __attribute__((overloadable)) sub_group_scan_exclusive_add(short x);
    short    __attribute__((overloadable)) sub_group_scan_exclusive_min(short x);
    short    __attribute__((overloadable)) sub_group_scan_exclusive_max(short x);
    short    __attribute__((overloadable)) sub_group_scan_inclusive_add(short x);
    short    __attribute__((overloadable)) sub_group_scan_inclusive_min(short x);
    short    __attribute__((overloadable)) sub_group_scan_inclusive_max(short x);

    ushort    __attribute__((overloadable)) sub_group_reduce_add(ushort x);
    ushort    __attribute__((overloadable)) sub_group_reduce_min(ushort x);
    ushort    __attribute__((overloadable)) sub_group_reduce_max(ushort x);
    ushort    __attribute__((overloadable)) sub_group_scan_exclusive_add(ushort x);
    ushort    __attribute__((overloadable)) sub_group_scan_exclusive_min(ushort x);
    ushort    __attribute__((overloadable)) sub_group_scan_exclusive_max(ushort x);
    ushort    __attribute__((overloadable)) sub_group_scan_inclusive_add(ushort x);
    ushort    __attribute__((overloadable)) sub_group_scan_inclusive_min(ushort x);
    ushort    __attribute__((overloadable)) sub_group_scan_inclusive_max(ushort x);
#endif // defined(cl_khr_subgroup_extended_types)

#if defined(cl_khr_subgroup_non_uniform_arithmetic) || defined(cl_khr_subgroup_clustered_reduce)
    #define DECL_SUB_GROUP_NON_UNIFORM_OPERATION(TYPE, GROUP_TYPE, OPERATION)
    #define DECL_SUB_GROUP_NON_UNIFORM_CLUSTERED_OPERATION(TYPE, GROUP_TYPE, OPERATION)

#if defined(cl_khr_subgroup_non_uniform_arithmetic)
    #define DECL_SUB_GROUP_NON_UNIFORM_OPERATION(TYPE, GROUP_TYPE, OPERATION) \
    TYPE    __attribute__((overloadable)) sub_group_non_uniform_##GROUP_TYPE##_##OPERATION(TYPE value);
#endif // defined(cl_khr_subgroup_non_uniform_arithmetic)

#if defined(cl_khr_subgroup_clustered_reduce)
    #define DECL_SUB_GROUP_NON_UNIFORM_CLUSTERED_OPERATION(TYPE, GROUP_TYPE, OPERATION) \
    TYPE    __attribute__((overloadable)) sub_group_clustered_##GROUP_TYPE##_##OPERATION(TYPE value, uint clustersize);
#endif // defined(cl_khr_subgroup_clustered_reduce)

    #define DECL_SUB_GROUP_NON_UNIFORM_ALL_GROUPS(TYPE, OPERATION)           \
    DECL_SUB_GROUP_NON_UNIFORM_OPERATION(TYPE, reduce, OPERATION)            \
    DECL_SUB_GROUP_NON_UNIFORM_OPERATION(TYPE, scan_inclusive, OPERATION)    \
    DECL_SUB_GROUP_NON_UNIFORM_OPERATION(TYPE, scan_exclusive, OPERATION)    \
    DECL_SUB_GROUP_NON_UNIFORM_CLUSTERED_OPERATION(TYPE, reduce, OPERATION)

    // ARITHMETIC OPERATIONS
    // gentype sub_group_non_uniform_GROUP_TYPE_add(gentype value)
    // gentype sub_group_non_uniform_GROUP_TYPE_min(gentype value)
    // gentype sub_group_non_uniform_GROUP_TYPE_max(gentype value)
    // gentype sub_group_non_uniform_GROUP_TYPE_mul(gentype value)
    #define DECL_SUB_GROUP_ARITHMETIC_OPERATIONS(TYPE)   \
    DECL_SUB_GROUP_NON_UNIFORM_ALL_GROUPS(TYPE, add)     \
    DECL_SUB_GROUP_NON_UNIFORM_ALL_GROUPS(TYPE, min)     \
    DECL_SUB_GROUP_NON_UNIFORM_ALL_GROUPS(TYPE, max)     \
    DECL_SUB_GROUP_NON_UNIFORM_ALL_GROUPS(TYPE, mul)

    DECL_SUB_GROUP_ARITHMETIC_OPERATIONS(char)
    DECL_SUB_GROUP_ARITHMETIC_OPERATIONS(uchar)
    DECL_SUB_GROUP_ARITHMETIC_OPERATIONS(short)
    DECL_SUB_GROUP_ARITHMETIC_OPERATIONS(ushort)
    DECL_SUB_GROUP_ARITHMETIC_OPERATIONS(int)
    DECL_SUB_GROUP_ARITHMETIC_OPERATIONS(uint)
    DECL_SUB_GROUP_ARITHMETIC_OPERATIONS(long)
    DECL_SUB_GROUP_ARITHMETIC_OPERATIONS(ulong)
    DECL_SUB_GROUP_ARITHMETIC_OPERATIONS(float)
    #if defined(cl_khr_fp64)
    DECL_SUB_GROUP_ARITHMETIC_OPERATIONS(double)
    #endif // defined(cl_khr_fp64)
    #if defined(cl_khr_fp16)
    DECL_SUB_GROUP_ARITHMETIC_OPERATIONS(half)
    #endif // defined(cl_khr_fp16)

    // BITWISE OPERATIONS
    // gentype sub_group_non_uniform_GROUP_TYPE_and(gentype value)
    // gentype sub_group_non_uniform_GROUP_TYPE_or(gentype value)
    // gentype sub_group_non_uniform_GROUP_TYPE_xor(gentype value)
    #define DECL_SUB_GROUP_BITWISE_OPERATIONS(TYPE)    \
    DECL_SUB_GROUP_NON_UNIFORM_ALL_GROUPS(TYPE, and)   \
    DECL_SUB_GROUP_NON_UNIFORM_ALL_GROUPS(TYPE, or)    \
    DECL_SUB_GROUP_NON_UNIFORM_ALL_GROUPS(TYPE, xor)

    DECL_SUB_GROUP_BITWISE_OPERATIONS(char)
    DECL_SUB_GROUP_BITWISE_OPERATIONS(uchar)
    DECL_SUB_GROUP_BITWISE_OPERATIONS(short)
    DECL_SUB_GROUP_BITWISE_OPERATIONS(ushort)
    DECL_SUB_GROUP_BITWISE_OPERATIONS(int)
    DECL_SUB_GROUP_BITWISE_OPERATIONS(uint)
    DECL_SUB_GROUP_BITWISE_OPERATIONS(long)
    DECL_SUB_GROUP_BITWISE_OPERATIONS(ulong)

    // LOGICAL OPERATIONS
    // int sub_group_non_uniform_GROUP_TYPE_logical_and(int predicate)
    // int sub_group_non_uniform_GROUP_TYPE_logical_or(int predicate)
    // int sub_group_non_uniform_GROUP_TYPE_logical_xor(int predicate)
    #define DECL_SUB_GROUP_BITWISE_OPERATIONS(TYPE)             \
    DECL_SUB_GROUP_NON_UNIFORM_ALL_GROUPS(TYPE, logical_and)    \
    DECL_SUB_GROUP_NON_UNIFORM_ALL_GROUPS(TYPE, logical_or)     \
    DECL_SUB_GROUP_NON_UNIFORM_ALL_GROUPS(TYPE, logical_xor)

    DECL_SUB_GROUP_BITWISE_OPERATIONS(int)

#endif // defined(cl_khr_subgroup_non_uniform_arithmetic) || defined(cl_khr_subgroup_clustered_reduce)

)CLC";