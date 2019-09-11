kernel void Test( global uint* dst )
{
    uint index = get_global_id(0);
    dst[index] = index;
}
