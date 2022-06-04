/*
// Copyright (c) 2021 Ben Ashbaugh
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
*/
#pragma once

#include <CL/cl_layer.h>

#include <vector>

template<class T>
cl_int writeParamToMemory(
    size_t param_value_size,
    T param,
    size_t* param_value_size_ret,
    T* pointer)
{
    if (pointer != nullptr) {
        if (param_value_size < sizeof(param)) {
            return CL_INVALID_VALUE;
        }
        *pointer = param;
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = sizeof(param);
    }

    return CL_SUCCESS;
}

template< class T >
cl_int writeVectorToMemory(
    size_t param_value_size,
    const std::vector<T>& param,
    size_t *param_value_size_ret,
    T* pointer )
{
    cl_int  errorCode = CL_SUCCESS;

    size_t  size = param.size() * sizeof(T);

    if( pointer != NULL )
    {
        if( param_value_size < size )
        {
            errorCode = CL_INVALID_VALUE;
        }
        else
        {
            memcpy(pointer, param.data(), size);
        }
    }

    if( param_value_size_ret != NULL )
    {
        *param_value_size_ret = size;
    }

    return errorCode;
}


