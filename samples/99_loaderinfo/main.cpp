/*
// Copyright (c) 2019-2020 Ben Ashbaugh
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

#include <stdio.h>
#include <vector>
#include <popl/popl.hpp>

#include <CL/cl.h>

typedef cl_uint cl_icdl_info;

#define CL_ICDL_OCL_VERSION 1
#define CL_ICDL_VERSION     2
#define CL_ICDL_NAME        3
#define CL_ICDL_VENDOR      4

typedef cl_int (*pfn_clGetICDLoaderInfoOCLICD)(cl_icdl_info, size_t, void*, size_t*);
pfn_clGetICDLoaderInfoOCLICD clGetICDLoaderInfoOCLICD = NULL;

static void PrintLoaderInfo(const char* label, cl_icdl_info info)
{
    size_t sz = 0;
    clGetICDLoaderInfoOCLICD(info, 0, nullptr, &sz);

    std::vector<char> str(sz);
    clGetICDLoaderInfoOCLICD(info, sz, str.data(), nullptr);

    printf("Query for for %s (size = %zu) returned: %s\n", label, sz, str.data());
}

int main(
    int argc,
    char** argv )
{
    {
        popl::OptionParser op("Supported Options");

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: loaderinfo [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }


    clGetICDLoaderInfoOCLICD = (pfn_clGetICDLoaderInfoOCLICD)
        clGetExtensionFunctionAddress("clGetICDLoaderInfoOCLICD");

    if (clGetICDLoaderInfoOCLICD == NULL) {
        printf("Couldn't get function pointer to clGetICDLoaderInfoOCLICD!\n");
        printf("This is normal and some ICD loaders do not support this functionality.\n");
        printf("Exiting...\n");
        return 0;
    }

    #define QUERY_AND_PRINT_LOADER_INFO(_info)  \
        PrintLoaderInfo(#_info, _info);

    QUERY_AND_PRINT_LOADER_INFO(CL_ICDL_OCL_VERSION);
    QUERY_AND_PRINT_LOADER_INFO(CL_ICDL_VERSION);
    QUERY_AND_PRINT_LOADER_INFO(CL_ICDL_NAME);
    QUERY_AND_PRINT_LOADER_INFO(CL_ICDL_VENDOR);

    return 0;
}
