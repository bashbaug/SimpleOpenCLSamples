/*
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

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <assert.h>
#include <chrono>

#include <linux/limits.h>
#include <libgen.h>
#include <unistd.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

/* Compute c = a + b.

Derived from https://gist.github.com/ddemidov/2925717 with various fixes and tweaks:

- Add event handler and wait statement to prevent race conditions
- Move OpenCL kernel to separate source file for readability
- Use enqueueNDRangeKernel arguments instead of kernel conditional to control work item count
- Query dimension information from device
- Timing information
- Code formatting
*/

int main()
{
	try
	{
		// Get list of OpenCL platforms.
		std::vector<cl::Platform> platform;
		cl::Platform::get(&platform);

		if (platform.empty())
		{
			std::cerr << "OpenCL platforms not found." << std::endl;
			return 1;
		}

		// Get first available GPU device which supports double precision.
		cl::Context context;
		std::vector<cl::Device> devices;
		for (auto p = platform.begin(); devices.empty() && p != platform.end(); p++)
		{
			std::vector<cl::Device> pldev;

			try
			{
				p->getDevices(CL_DEVICE_TYPE_GPU, &pldev);

				for (auto d = pldev.begin(); devices.empty() && d != pldev.end(); d++)
				{
					if (!d->getInfo<CL_DEVICE_AVAILABLE>())
						continue;

					std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();

					if (
						ext.find("cl_khr_fp64") == std::string::npos &&
						ext.find("cl_amd_fp64") == std::string::npos)
						continue;

					devices.push_back(*d);
					context = cl::Context(devices);
				}
			}
			catch (...)
			{
				devices.clear();
			}
		}

		if (devices.empty())
		{
			std::cerr << "GPUs with double precision not found." << std::endl;
			return 1;
		}

		std::cout << "Using device " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;

		auto dimensions = devices[0].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

		std::cout << "Max dimensions: " << dimensions[0] << "x" << dimensions[1] << "x" << dimensions[2] << std::endl;
		size_t global_dim = dimensions[0] * dimensions[1] * dimensions[2];

		// Create command queue.
		cl::CommandQueue queue(context, devices[0]);

		// Compile OpenCL program for found device.

		// get path to CL file
		char result[PATH_MAX];
		ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
		const char *path;
		if (count != -1) {
			path = dirname(result);
		}
		std::string kernel_path = std::string(path) + "/hello.cl";
		std::ifstream source_file(kernel_path);
		if (!source_file.good()) {
			std::cout << "Failed to find kernel file " << kernel_path << std::endl;
			return 2;
		}

		// read CL file and feed it to the online compiler
		std::string source_code(
			std::istreambuf_iterator<char>(source_file),
			(std::istreambuf_iterator<char>()));
		cl::Program program(
			context,
			cl::Program::Sources(
				1, std::make_pair(source_code.c_str(), source_code.length())));

		try
		{
			program.build(devices);
		}
		catch (const cl::Error &)
		{
			std::cerr
				<< "OpenCL compilation error" << std::endl
				<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
				<< std::endl;
			return 1;
		}

		cl::Kernel hello(program, "hello_kernel");

		// Prepare input data.
		std::vector<double> a(global_dim, 1);
		std::vector<double> b(global_dim, 2);
		std::vector<double> c(global_dim);

		// Allocate device buffers and transfer input data to device.
		cl::Buffer A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					 a.size() * sizeof(double), a.data());

		cl::Buffer B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					 b.size() * sizeof(double), b.data());

		cl::Buffer C(context, CL_MEM_READ_WRITE,
					 c.size() * sizeof(double));

		// Set kernel parameters.
		hello.setArg(0, A);
		hello.setArg(1, B);
		hello.setArg(2, C);

		// Setup event handler
		cl::Event *event_handler = new cl::Event();

		auto begin = std::chrono::high_resolution_clock::now();

		// Launch kernel on the compute device.
		queue.enqueueNDRangeKernel(
			hello,
			cl::NullRange,
			cl::NDRange(
				global_dim,
				1,
				1),
			cl::NullRange,
			nullptr, event_handler);

		// Wait for kernel to complete so we get accurate timing
		event_handler->wait();

		auto end = std::chrono::high_resolution_clock::now();
		std::cout
			<< "Computed " << global_dim << " values in "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
			<< "ms" << std::endl;

		// Get result back to host.
		queue.enqueueReadBuffer(C, CL_TRUE, 0, c.size() * sizeof(double), c.data(), nullptr, event_handler);

		// Wait for read back to complete before verifying results
		event_handler->wait();

		delete event_handler;

		// Should get <global_dim> number of lines, each with result 3
		for (uint i = 0; i < global_dim; ++i)
		{
			assert(c[i] == 3);
		}
	}
	catch (const cl::Error &err)
	{
		std::cerr
			<< "OpenCL error: "
			<< err.what() << "(" << err.err() << ")"
			<< std::endl;
		return 1;
	}
}
