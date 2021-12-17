kernel void Test( global uint* dst )
{
    size_t index = get_global_id(0);
    if (index % 1024 == 0) {
        atomic_inc(dst);
    }
}
