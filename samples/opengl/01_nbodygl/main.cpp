/*
// Copyright (c) 2022 Ben Ashbaugh
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

#include <popl/popl.hpp>

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/opencl.hpp>

#include <GLFW/glfw3.h>
#if defined(WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#else
#error Unknown OS!
#endif
#include <GLFW/glfw3native.h>

#include "util.hpp"

#include <chrono>
#include <random>
#include <math.h>

GLFWwindow* pWindow = NULL;

bool use_cl_khr_gl_sharing = true;
bool use_cl_khr_gl_event = true;
bool animate = false;
bool redraw = false;
bool vsync = true;

size_t width = 1024;
size_t height = 1024;

size_t numBodies = 1024;
size_t groupSize = 0;

cl::Context context;
cl::CommandQueue commandQueue;
cl::Kernel kernel;
cl::Buffer current_pos;
cl::Buffer current_vel;
cl::Buffer next_pos;
cl::Buffer next_vel;

static const char kernelString[] = R"CLC(
__kernel void nbody_step(
    const __global float4* pos,
    const __global float4* vel,
    __global float4* newPosition,
    __global float4* newVelocity)
{
    const uint numBodies = get_global_size(0);
    const float G = 1.0f / numBodies;
    const float dampen = 0.90f;
    const float deltaTime = 0.005f;
    const float epsilon = 1e-3;

    float3 myPos = pos[get_global_id(0)].xyz;
    float myMass = pos[get_global_id(0)].w;

    float3 myAcc = 0.0f;

    for(uint j = 0; j < numBodies; j++)
    {
        float3 otherPos = pos[j].xyz;
        float otherMass = pos[j].w;

        float3 deltaPos = otherPos - myPos;
        float r = fast_length(deltaPos) + epsilon;
        float a = G * otherMass / (r * r);

        myAcc += a * deltaPos / r;
    }

    float3 myVel = vel[ get_global_id(0) ].xyz;

    float4 newPos;
    newPos.xyz = myPos + myVel * deltaTime;
    newPos.w = myMass;

    float4 newVel;
    newVel.xyz = myVel + myAcc * deltaTime;
    newVel.w = 0;

    newVel *= dampen;

    newPosition[ get_global_id(0) ] = newPos;
    newVelocity[ get_global_id(0) ] = newVel;
}
)CLC";

// This function determines if the platform and device support CL-GL sharing
// extensions, and if so, create a context supporting sharing.  This requires
// three steps:
//      1. Querying the device to ensure the extensions are supported.
//      2. Querying devices that can interoperate with the OpenGL context.
//      3. If both queries are successful, creating an OpenCL context with the
//         OpenGL context.
// If any of these steps fail or if sharing is disabled then an OpenCL context
// is created that does not support sharing.
cl::Context createContext(const cl::Platform& platform, const cl::Device& device)
{
    const cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform(),
#if defined(WIN32)
        CL_GL_CONTEXT_KHR, (cl_context_properties)glfwGetWGLContext(pWindow),
        CL_WGL_HDC_KHR, (cl_context_properties)GetDC(glfwGetWin32Window(pWindow)),
#elif defined(__linux__)
        CL_GL_CONTEXT_KHR, (cl_context_properties)glfwGetGLXContext(pWindow),
        CL_GLX_DISPLAY_KHR, (cl_context_properties)glfwGetX11Display(),
#else
#error Unknown OS!
#endif
        0
    };

    if (checkDeviceForExtension(device, "cl_khr_gl_sharing")) {
        printf("Device supports cl_khr_gl_sharing.\n");
    } else {
        printf("Device does not support cl_khr_gl_sharing.\n");
        use_cl_khr_gl_sharing = false;
        use_cl_khr_gl_event = false;
    }

    if (checkDeviceForExtension(device, "cl_khr_gl_event")) {
        printf("Device supports cl_khr_gl_event.\n");
    } else {
        printf("Device does not support cl_khr_gl_event.\n");
        use_cl_khr_gl_event = false;
    }

    if (use_cl_khr_gl_sharing) {
        bool found = false;

        auto clGetGLContextInfoKHR =
            (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddressForPlatform(
                platform(),
                "clGetGLContextInfoKHR");
        if (clGetGLContextInfoKHR) {
            size_t sz = 0;

            clGetGLContextInfoKHR(
                props,
                CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,
                0,
                NULL,
                &sz);
            if (sz) {
                std::vector<cl_device_id> devices(sz / sizeof(cl_device_id));
                clGetGLContextInfoKHR(
                    props,
                    CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,
                    sz,
                    devices.data(),
                    NULL);
                printf("\nOpenCL Devices currently associated with the OpenGL Context:\n");
                for (auto& check : devices) {
                    found |= check == device();
                    printf("  %s\n", cl::Device(check).getInfo<CL_DEVICE_NAME>().c_str());
                }
            }

            clGetGLContextInfoKHR(
                props,
                CL_DEVICES_FOR_GL_CONTEXT_KHR,
                0,
                NULL,
                &sz);
            if (sz) {
                std::vector<cl_device_id> devices(sz / sizeof(cl_device_id));
                clGetGLContextInfoKHR(
                    props,
                    CL_DEVICES_FOR_GL_CONTEXT_KHR,
                    sz,
                    devices.data(),
                    NULL);
                printf("\nOpenCL Devices which may be associated with the OpenGL Context:\n");
                for (auto& check : devices) {
                    found |= check == device();
                    printf("  %s\n", cl::Device(check).getInfo<CL_DEVICE_NAME>().c_str());
                }
            }
        }

        if (found) {
            printf("Requested OpenCL device can share with the OpenGL context.\n");
        } else {
            printf("Requested OpenCL device cannot share with the OpenGL context.\n");
            use_cl_khr_gl_sharing = false;
        }
    }

    if (use_cl_khr_gl_sharing) {
        printf("Creating a context with GL sharing.\n");
        return cl::Context(device, props);
    }

    printf("Creating a context without GL sharing.\n");
    return cl::Context(device);
}

#if 0
// This function sets up an OpenGL texture with the requested dimensions.  If
// CL-GL sharing is supported and enabled, an OpenCL image is created from the
// OpenGL texture.  Otherwise, a standard OpenCL image is created, and the
// contents of the image will need to be copied to the OpenGL texture.
cl::Image2D createImage(const cl::Context& context)
{
    GLuint texname = 0;
    glGenTextures(1, &texname);
    glBindTexture(GL_TEXTURE_2D, texname);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        (GLsizei)gwx, (GLsizei)gwy,
        0,
        GL_RGBA,
        GL_FLOAT,
        NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glEnable( GL_TEXTURE_2D );

    GLenum  gl_error = glGetError();
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "Error: OpenGL generated %x!\n", gl_error);
    }

    if (use_cl_khr_gl_sharing) {
        // Note: clCreateFromGLTexture2D is an extension API, but it is
        // exported directly from the ICD loader.
        return cl::Image2D{
            clCreateFromGLTexture2D(
                context(),
                CL_MEM_WRITE_ONLY,
                GL_TEXTURE_2D,
                0,
                texname,
                NULL)};
    }

    return cl::Image2D{
        context,
        CL_MEM_WRITE_ONLY,
        cl::ImageFormat{CL_RGBA, CL_FLOAT},
        gwx, gwy };
}
#endif

static void init()
{
    std::mt19937 gen;
    std::uniform_real_distribution<float> rand_pos(-0.01f, 0.01f);
    std::uniform_real_distribution<float> rand_mass(0.1f, 1.0f);

    std::vector<cl_float4> init_pos(numBodies);
    std::vector<cl_float4> init_vel(numBodies);

    for (size_t i = 0; i < numBodies; i++) {
        // X, Y, and Z position:
        init_pos[i].s[0] = rand_pos(gen);
        init_pos[i].s[1] = rand_pos(gen);
        init_pos[i].s[2] = rand_pos(gen);

        // Mass:
        init_pos[i].s[3] = rand_mass(gen);

        // Initial velocity is zero:
        init_vel[i].s[0] = 0.0f;
        init_vel[i].s[1] = 0.0f;
        init_vel[i].s[2] = 0.0f;
        init_vel[i].s[3] = 0.0f;
    }

    current_pos = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, numBodies * sizeof(cl_float4), init_pos.data());
    current_vel = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, numBodies * sizeof(cl_float4), init_vel.data());

    next_pos = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, numBodies * sizeof(cl_float4));
    next_vel = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, numBodies * sizeof(cl_float4));
}

static void resize(GLFWwindow* pWindow, int width, int height)
{
    glViewport(0, 0, width, height);
}

static void display(void)
{
    if (animate) {
        static size_t startFrame = 0;
        static size_t frame = 0;
        static std::chrono::system_clock::time_point start =
            std::chrono::system_clock::now();

        ++frame;

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<float> delta = end - start;
        float elapsed_seconds = delta.count();
        if (elapsed_seconds > 2.0f) {
            printf("FPS: %.1f\n", (frame - startFrame) / elapsed_seconds);
            startFrame = frame;
            start = end;
        }
    }
    if (redraw) {
        redraw = false;
    }

#if 0
    // If we support interop we need to acquire the OpenCL image object we
    // created from the OpenGL texture.  If we do not support interop then we
    // will compute into the OpenCL image object then manually transfer its
    // contents to OpenGL.
    if (use_cl_khr_gl_sharing) {
        // If we do not support cl_khr_gl_event then we need to synchronize
        // OpenGL and OpenCL.  If we do support cl_khr_gl_event, then acquiring
        // the object performs an implicit synchronization.
        if (use_cl_khr_gl_event == false) {
            glFinish();
        }
        clEnqueueAcquireGLObjects(
            commandQueue(),
            1,
            &mem(),
            0,
            NULL,
            NULL);
    }
#endif

    kernel.setArg(0, current_pos);
    kernel.setArg(1, current_vel);
    kernel.setArg(2, next_pos);
    kernel.setArg(3, next_vel);

    cl::NDRange lws;    // NullRange by default.
    if( groupSize > 0 )
    {
        lws = cl::NDRange{groupSize};
    }

    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{numBodies},
        lws);

    // After executing the OpenCL kernel, we need to release the OpenCL image
    // object back to OpenGL, or manually copy from OpenCL to OpenGL.
#if 0
    if (use_cl_khr_gl_sharing) {
        // As before, synchronize if we do not support cl_khr_gl_event.
        if (use_cl_khr_gl_event == false) {
            commandQueue.finish();
        }
        clEnqueueReleaseGLObjects(
            commandQueue(),
            1,
            &mem(),
            0,
            NULL,
            NULL);
    } else {
        // For the manual copy, we will map the OpenCL image object, transfer
        // its contents to OpenGL, then unmap the OpenCL image object.
        size_t rowPitch = 0;
        void* pixels = commandQueue.enqueueMapImage(
            mem,
            CL_TRUE,
            CL_MAP_READ,
            {0, 0, 0},
            {gwx, gwy, 1},
            &rowPitch,
            nullptr);

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            (GLsizei)gwx, (GLsizei)gwy,
            0,
            GL_RGBA,
            GL_FLOAT,
            pixels);

        commandQueue.enqueueUnmapMemObject(
            mem,
            pixels);
    }
#endif

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);

    glColor3f(1.0f, 0.6f, 0.0f);

    const cl_float4* p = (const cl_float4*)commandQueue.enqueueMapBuffer(current_pos, CL_TRUE, CL_MAP_READ, 0, numBodies * sizeof(cl_float4));

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, sizeof(cl_float4), p);
    glDrawArrays(GL_POINTS, 0, (GLsizei)numBodies);
    glDisableClientState(GL_VERTEX_ARRAY);

    commandQueue.enqueueUnmapMemObject(current_pos, (void*)p);

    std::swap(current_pos, next_pos);
    std::swap(current_vel, next_vel);

    GLenum  gl_error = glGetError();
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "Error: OpenGL generated %x!\n", gl_error);
    }

    glfwSwapBuffers(pWindow);
}

static void keyboard(GLFWwindow* pWindow, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(pWindow, GLFW_TRUE);
            break;
        case GLFW_KEY_SPACE:
            animate = !animate;
            printf("animation is %s\n", animate ? "ON" : "OFF");
            break;

        case GLFW_KEY_S:
            printf("stepping...\n");
            redraw = true;
            break;
        case GLFW_KEY_R:
            printf("reinitializing...\n");
            init();
            break;

        case GLFW_KEY_V:
            vsync = !vsync;
            printf("vsync is %s\n", vsync ? "ON" : "OFF");
            if (vsync) {
                glfwSwapInterval(1);
            } else {
                glfwSwapInterval(0);
            }
            break;
        }
    }
}

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    {
        bool hostCopy = false;
        bool hostSync = false;

        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Switch>("", "hostcopy", "Do not use cl_khr_gl_sharing", &hostCopy);
        op.add<popl::Switch>("", "hostsync", "Do not use cl_khr_gl_event", &hostSync);
        op.add<popl::Value<size_t>>("n", "numbodies", "Number of Bodies", numBodies, &numBodies);
        op.add<popl::Value<size_t>>("g", "groupsize", "Group Size", groupSize, &groupSize);
        op.add<popl::Value<size_t>>("w", "width", "Render Width", width, &width);
        op.add<popl::Value<size_t>>("h", "height", "Render Height", height, &height);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: nbodygl [options]\n"
                "%s", op.help().c_str());
            return -1;
        }

        use_cl_khr_gl_sharing = !hostCopy;
        use_cl_khr_gl_event = !hostSync;
    }

    if (!glfwInit()) {
        fprintf(stderr, "Could not initialize glfw!\n");
        return -1;
    }

    // Create an OpenGL window.  This needs to be done before creating the
    // OpenCL context because we need to know information about the OpenGL
    // context to create an OpenCL context that supports sharing.
    pWindow = glfwCreateWindow((int)width, (int)height, "N-Body Simulation with OpenGL", NULL, NULL);
    glfwMakeContextCurrent(pWindow);

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    context = createContext(platforms[platformIndex], devices[deviceIndex]);
    commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
    kernel = cl::Kernel{ program, "nbody_step" };

    init();

    glfwSetKeyCallback(pWindow, keyboard);
    glfwSetFramebufferSizeCallback(pWindow, resize);

    while (!glfwWindowShouldClose(pWindow)) {
        if (animate || redraw) {
            display();
        }

        glfwPollEvents();
    }

    glfwDestroyWindow(pWindow);
    glfwTerminate();

    return 0;
}
