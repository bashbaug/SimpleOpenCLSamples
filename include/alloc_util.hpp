/*
// Copyright (c) 2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/
#pragma once

#if defined(_WIN32)
#include <cstdlib>
#endif // defined(_WIN32)

#if defined(__linux__) || defined(linux) || defined(__APPLE__)
#if defined(__ANDROID__)
#include <malloc.h>
#else
#include <stdlib.h>
#endif // defined(__ANDROID__)
#endif // defined(__linux__) || defined(linux) || defined(__APPLE__)

#if defined(__MINGW32__)
#include <malloc.h>
#if defined(__MINGW64__)
// mingw-w64 doesnot have __mingw_aligned_malloc, instead it has _aligned_malloc
#define __mingw_aligned_malloc _aligned_malloc
#define __mingw_aligned_free _aligned_free
#endif // defined(__MINGW64__)
#endif // defined(__MINGW32__)

static inline void* align_malloc(size_t size, size_t alignment)
{
#if defined(_WIN32) && defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#elif defined(__linux__) || defined(linux) || defined(__APPLE__)
#if defined(__ANDROID__)
    return memalign(alignment, size);
#else
    alignment = (alignment < sizeof(void*)) ? sizeof(void*) : alignment;
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) == 0) {
        return ptr;
    } else {
        return nullptr;
    }
#endif
#elif defined(__MINGW32__)
    return __mingw_aligned_malloc(size, alignment);
#else
#error "Please add align_malloc implementation."
    return nullptr;
#endif
}

static inline void align_free(void* ptr)
{
#if defined(_WIN32) && defined(_MSC_VER)
    _aligned_free(ptr);
#elif defined(__linux__) || defined(linux) || defined(__APPLE__)
    free(ptr);
#elif defined(__MINGW32__)
    __mingw_aligned_free(ptr);
#else
#error "Please add align_free implementation."
#endif
}
