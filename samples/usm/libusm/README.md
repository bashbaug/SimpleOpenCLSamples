# libusm

Because extension APIs are not exported by the OpenCL ICD loader most applications link to, the extension APIs must be defined by some other mechanism.

The `libusm` library provides definitions of the extension APIs for `cl_intel_unified_shared_memory` and an initialization function to get the Unified Shared Memory extension APIs for a specific platform.

## How to Use libusm

Using this library is intended to be straightforward:

* Link your application with the library.
* Before calling any Unified Shared Memory API functions, call the `libusm` initialization function to query the extension APIs for the specified platform.
To do this, your application will need to #include `libusm.h`.
* Calling any Unified Shared Memory extension API before initialization should not crash, but will return an OpenCL error code that must be handled appropriately.
* Likewise, if `libusm` is initialized with a platform that does not support some or all Unified Shared Memory extension APIs, calling an unsupported extension API will result an OpenCL error that must be handled appropriately.

`libusm` is in sync with the [USM extension draft](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/cl_intel_unified_shared_memory.asciidoc) revision H.

## Limitations of libusm

`libusm` has limitations and is primarily intended for the Unified Shared Memory samples.
If you want to use `libusm` in another software product, please be aware of the following limitations:

* `libusm` supports a single platform only.
Calling Unified Shared Memory extension APIs from a single application into two different platforms, such as a CPU platform and a GPU platform, is not currently supported.
* `libusm` is not thread-safe.
Care should be taken when calling into the `libusm` initialization function from multiple threads.
* `libusm` only supports the Unified Shared Memory extension APIs.
Other extension APIs are not currently supported.
