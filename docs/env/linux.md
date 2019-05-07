# Verify Hardware Support

Most modern GPUs support OpenCL. For integrated graphics devices (iGPUs), use `lscpu` to get the processor SKU. Detailed information for Intel SKUs is available from [ark.intel.com](ark.intel.com). Detailed information for AMD processors is available from [AMD's product page](https://www.amd.com/en/products/specifications/processors).

# Build Dependencies

Building OpenCL applications requires headers for the OpenCL API and OpenCL ICD Loader. These headers should be provided automatically by CMake, but can also be manually imported as follows.

```
$ git submodule update --init -- external/OpenCL-Headers
$ git submodule update --init -- external/opencl-icd-loader
```

# Runtime Dependencies

To run the samples, you will need to obtain and install an ICD loader and an
OpenCL implementation (ICD) that supports the `cl_khr_icd` extension.

The ICD loader is likely provided by your operating system or an OpenCL
implementation.  If desired, you may use the ICD loader that is built along
with these OpenCL samples.  The OpenCL implementation will likely be provided
by your OpenCL device vendor.  There are several open source OpenCL
implementations as well.