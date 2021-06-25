kernel void WriteConstant(global uint* dst)
{
    int id = get_global_id(0);
    dst[id] = 42;
}
