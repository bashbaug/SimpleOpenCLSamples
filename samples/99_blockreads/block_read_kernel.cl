#if !defined(cl_intel_subgroup_2d_block_io)
#error cl_intel_subgroup_2d_block_io is not supported!
#endif

ushort2  __builtin_IB_subgroup_block_read_flat_u16_m2k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);

__attribute__((intel_reqd_sub_group_size(16)))
kernel void BlockReadTest(global ushort* matrix, int width, int height)
{
    int2 coord = (int2)(0, 0);
    int bytewidth = width * sizeof(ushort);
    int bytepitch = bytewidth;
#if 0
    // This is the most basic 2D block read.
    // Each work-item gets one 16-bit value, from a single row.
    ushort data[1];
    intel_sub_group_2d_block_read_16b_1r16x1c(matrix, bytewidth, height, bytepitch, coord, data);
    printf("GID %3d: data = %04X\n", (int)get_global_id(0),
        data[0]);
#elif 0
    // This is a multi-row 2D block read.
    // Each work-item gets two 16-bit values, one from the first row, and one from the second row.
    ushort data[2];
    intel_sub_group_2d_block_read_16b_2r16x1c(matrix, bytewidth, height, bytepitch, coord, data);
    printf("GID %3d: data = %04X %04X\n", (int)get_global_id(0),
        data[0], data[1]);
#elif 0
    // This is another multi-row 2D block read.
    // Each work-item gets four 16-bit values, one from the first row, and one from the second row, etc.
    // Each work-item therefore gets four rows of data from the same matrix column.
    ushort data[4];
    intel_sub_group_2d_block_read_16b_4r16x1c(matrix, bytewidth, height, bytepitch, coord, data);
    printf("GID %3d: data = %04X %04X %04X %04X\n", (int)get_global_id(0),
        data[0], data[1], data[2], data[3]);
#elif 0
    // This is the most basic transposed 2D block read, given that we have not implemented a single-column transposed block read.
    // Each work-item gets eight 32-bit values, where each 32-bit value contains two columns of data (pre-transpose).
    // Each work-item therefore gets 16 columns of data from the same matrix row.
    uint data[8];
    intel_sub_group_2d_block_read_transpose_32b_16r8x1c(matrix, bytewidth, height, bytepitch, coord, data);
    printf("GID %3d: data = %08X %08X %08X %08X %08X %08X %08X %08X ...\n", (int)get_global_id(0),
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
#elif 1
    // This is a more complicated transposed 2D block read, since there are 32 rows (pre-transpose) and only 16 work-items.
    // Each work-item gets 16 32-bit values, where each 32-bit value contains two columns of data (pre-transpose).
    // Each work-item therefore gets 16 columns of data from one matrix row, and 16 columns of data from another matrix row.
    // The data from the two matrix rows are interleaved, so there are two columns of data from one row, then two columns from the other row, etc.
    uint data[16];
    intel_sub_group_2d_block_read_transpose_32b_32r8x1c(matrix, bytewidth, height, bytepitch, coord, data);
    printf("GID %3d: data = %08X %08X %08X %08X %08X %08X %08X %08X ...\n", (int)get_global_id(0),
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
#endif
}
