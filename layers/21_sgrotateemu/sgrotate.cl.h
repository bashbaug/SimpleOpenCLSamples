const char* g_SubgroupRotateString = R"CLC(

#if !defined(cl_khr_subgroup_rotate)
#define cl_khr_subgroup_rotate
#endif

#define sub_group_rotate            __emu_sub_group_rotate
#define sub_group_clustered_rotate  __emu_sub_group_clustered_rotate

static inline uint __emu_rotate_index(int delta) {
    uint sglid = get_sub_group_local_id();
    uint mask = get_max_sub_group_size() - 1;
    return (sglid + delta) & mask;
}
static inline uint __emu_clustered_rotate_index(int delta, uint cluster_size) {
    uint sglid = get_sub_group_local_id();
    uint mask = cluster_size - 1;
    return ((sglid + delta) & mask) + (sglid & ~mask);
}

#define __EMU_SG_ROTATE_HELPER(_type)                                                                       \
_type __attribute__((overloadable)) sub_group_rotate(_type value, int delta) {                              \
    return sub_group_shuffle(value, __emu_rotate_index(delta));                                             \
}                                                                                                           \
_type __attribute__((overloadable)) sub_group_clustered_rotate(_type value, int delta, uint cluster_size) { \
    return sub_group_shuffle(value, __emu_clustered_rotate_index(delta, cluster_size));                     \
}

__EMU_SG_ROTATE_HELPER(char)
__EMU_SG_ROTATE_HELPER(uchar)
__EMU_SG_ROTATE_HELPER(short)
__EMU_SG_ROTATE_HELPER(ushort)
__EMU_SG_ROTATE_HELPER(int)
__EMU_SG_ROTATE_HELPER(uint)
__EMU_SG_ROTATE_HELPER(long)
__EMU_SG_ROTATE_HELPER(ulong)
__EMU_SG_ROTATE_HELPER(float)
#if defined(cl_khr_fp64)
__EMU_SG_ROTATE_HELPER(double)
#endif // defined(cl_khr_fp64)
#if defined(cl_khr_fp16)
__EMU_SG_ROTATE_HELPER(half)
#endif // defined(cl_khr_fp16)

)CLC";