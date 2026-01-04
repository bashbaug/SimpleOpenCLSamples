/*
// Copyright (c) 2023-2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
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
#include <math.h>

#if !defined(GL_CLAMP_TO_EDGE)
#define GL_CLAMP_TO_EDGE                  0x812F
#endif

GLFWwindow* pWindow = NULL;

bool use_cl_khr_gl_sharing = true;
bool use_cl_khr_gl_event = true;
bool animate = false;
bool redraw = false;
bool vsync = true;

size_t gwx = 512;
size_t gwy = 512;
size_t lwx = 0;
size_t lwy = 0;

float cr = -0.123f;
float ci =  0.745f;

cl::CommandQueue commandQueue;
cl::Kernel kernelJulia;
cl::Kernel kernelSobel;
cl::Sampler sampler;
cl::Image2D memTmp;
cl::Image2D memDst;

static const char kernelString[] = R"CLC(
kernel void Julia( write_only image2d_t dst, float cr, float ci )
{
    const float cMinX = -1.5f;
    const float cMaxX =  1.5f;
    const float cMinY = -1.5f;
    const float cMaxY =  1.5f;

    const int cWidth = get_global_size(0);
    const int cHeight = get_global_size(1);
    const int cIterations = 16;

    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);

    float a = x * ( cMaxX - cMinX ) / cWidth + cMinX;
    float b = y * ( cMaxY - cMinY ) / cHeight + cMinY;

    float result = 0.0f;
    const float thresholdSquared = cIterations * cIterations / 64.0f;

    for( int i = 0; i < cIterations; i++ ) {
        float aa = a * a;
        float bb = b * b;

        float magnitudeSquared = aa + bb;
        if( magnitudeSquared >= thresholdSquared ) {
            break;
        }

        result += 1.0f / cIterations;
        b = 2 * a * b + ci;
        a = aa - bb + cr;
    }

    result = max( result, 0.0f );
    result = min( result, 1.0f );

    // RGBA
    float4 color = (float4)( result, sqrt(result), 1.0f, 1.0f );

    write_imagef(dst, (int2)(x, y), color);
}

kernel void Sobel( read_only image2d_t src, write_only image2d_t dst, sampler_t sampler )
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);

    float4 texel_ul = read_imagef(src, sampler, (int2)(x - 1, y - 1));
    float4 texel_u  = read_imagef(src, sampler, (int2)(x    , y - 1));
    float4 texel_ur = read_imagef(src, sampler, (int2)(x + 1, y - 1));

    float4 texel_l  = read_imagef(src, sampler, (int2)(x - 1, y    ));
    float4 texel_r  = read_imagef(src, sampler, (int2)(x + 1, y    ));

    float4 texel_bl = read_imagef(src, sampler, (int2)(x - 1, y + 1));
    float4 texel_b  = read_imagef(src, sampler, (int2)(x    , y + 1));
    float4 texel_br = read_imagef(src, sampler, (int2)(x + 1, y + 1));

    float4 gx =
        texel_ul - texel_ur
        + 2.0f * texel_l - 2.0f * texel_r
        + texel_bl - texel_br;

    float4 gy =
        texel_ul + 2.0f * texel_u + texel_ur
        - texel_bl - 2.0f * texel_b - texel_br;

    float4 mag = sqrt(gx * gx + gy * gy);

    float grey = mag.x * 0.2126f + mag.y * 0.7152f + mag.z * 0.0722f;

    write_imagef(dst, (int2)(x, y), (float4)(grey, grey, grey, 1.0f));
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

        float fcr = (frame % 599) / 599.f * 2.0f * CL_M_PI_F;
        float fci = (frame % 773) / 773.f * 2.0f * CL_M_PI_F;
        cr = sinf(fcr);
        ci = sinf(fci);

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

    // Execute the Julia OpenCL kernel.
    // This kernel does not need OpenGL interop.
    kernelJulia.setArg(0, memTmp);
    kernelJulia.setArg(1, cr);
    kernelJulia.setArg(2, ci);

    cl::NDRange lws;    // NullRange by default.
    if( lwx > 0 && lwy > 0 )
    {
        lws = cl::NDRange{lwx, lwy};
    }

    commandQueue.enqueueNDRangeKernel(
        kernelJulia,
        cl::NullRange,
        cl::NDRange{gwx, gwy},
        lws);

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
            &memDst(),
            0,
            NULL,
            NULL);
    }

    // Execute the Sobel OpenCL kernel.
    kernelSobel.setArg(0, memTmp);
    kernelSobel.setArg(1, memDst);
    kernelSobel.setArg(2, sampler);

    commandQueue.enqueueNDRangeKernel(
        kernelSobel,
        cl::NullRange,
        cl::NDRange{gwx, gwy},
        lws);

    // After executing the OpenCL kernel, we need to release the OpenCL image
    // object back to OpenGL, or manually copy from OpenCL to OpenGL.
    if (use_cl_khr_gl_sharing) {
        // As before, synchronize if we do not support cl_khr_gl_event.
        if (use_cl_khr_gl_event == false) {
            commandQueue.finish();
        }
        clEnqueueReleaseGLObjects(
            commandQueue(),
            1,
            &memDst(),
            0,
            NULL,
            NULL);
    } else {
        // For the manual copy, we will map the OpenCL image object, transfer
        // its contents to OpenGL, then unmap the OpenCL image object.
        size_t rowPitch = 0;
        void* pixels = commandQueue.enqueueMapImage(
            memDst,
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
            memDst,
            pixels);
    }

    // Draw a triangle strip to cover the entire viewport.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBegin(GL_TRIANGLE_STRIP);
        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(-1.0f, -1.0);

        glTexCoord2f(0.0f, 1.0f);
        glVertex2f(-1.0f,  1.0f);

        glTexCoord2f(1.0f, 0.0f);
        glVertex2f( 1.0f, -1.0f);

        glTexCoord2f(1.0f, 1.0f);
        glVertex2f( 1.0f,  1.0f);
    glEnd();

    GLenum  gl_error = glGetError();
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "Error: OpenGL generated %x!\n", gl_error);
    }

    glfwSwapBuffers(pWindow);
}

static void keyboard(GLFWwindow* pWindow, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        redraw = true;

        switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(pWindow, GLFW_TRUE);
            break;
        case GLFW_KEY_SPACE:
            animate = !animate;
            printf("animation is %s\n", animate ? "ON" : "OFF");
            break;

        case GLFW_KEY_A:
            cr += 0.005f;
            break;
        case GLFW_KEY_Z:
            cr -= 0.005f;
            break;

        case GLFW_KEY_S:
            ci += 0.005f;
            break;
        case GLFW_KEY_X:
            ci -= 0.005f;
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
        bool paused = false;

        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Switch>("", "hostcopy", "Do not use cl_khr_gl_sharing", &hostCopy);
        op.add<popl::Switch>("", "hostsync", "Do not use cl_khr_gl_event", &hostSync);
        op.add<popl::Value<size_t>>("", "gwx", "Global Work Size X AKA Image Width", gwx, &gwx);
        op.add<popl::Value<size_t>>("", "gwy", "Global Work Size Y AKA Image Height", gwy, &gwy);
        op.add<popl::Value<size_t>>("", "lwx", "Local Work Size X", lwx, &lwx);
        op.add<popl::Value<size_t>>("", "lwy", "Local Work Size Y", lwy, &lwy);
        op.add<popl::Switch>("", "paused", "Start with Animation Paused", &paused);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: sobelgl [options]\n"
                "%s", op.help().c_str());
            return -1;
        }

        use_cl_khr_gl_sharing = !hostCopy;
        use_cl_khr_gl_event = !hostSync;
        animate = !paused;
    }

    if (!glfwInit()) {
        fprintf(stderr, "Could not initialize glfw!\n");
        return -1;
    }

    // Create an OpenGL window.  This needs to be done before creating the
    // OpenCL context because we need to know information about the OpenGL
    // context to create an OpenCL context that supports sharing.
    pWindow = glfwCreateWindow((int)gwx, (int)gwy, "Sobel Filter with OpenGL", NULL, NULL);
    glfwMakeContextCurrent(pWindow);

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (!checkPlatformIndex(platforms, platformIndex)) {
        return -1;
    }

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    cl::Context context = createContext(platforms[platformIndex], devices[deviceIndex]);
    commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
    kernelJulia = cl::Kernel{ program, "Julia" };
    kernelSobel = cl::Kernel{ program, "Sobel" };

    sampler = cl::Sampler{
        context,
        CL_FALSE,   // normalized coords
        CL_ADDRESS_CLAMP,
        CL_FILTER_NEAREST };

    memTmp = cl::Image2D{
        context,
        CL_MEM_READ_WRITE,
        cl::ImageFormat{CL_RGBA, CL_FLOAT},
        gwx, gwy };
    memDst = createImage(context);

    glfwSetKeyCallback(pWindow, keyboard);
    glfwSetFramebufferSizeCallback(pWindow, resize);

    while (!glfwWindowShouldClose(pWindow)) {
        if (animate || redraw) {
            display();
        }

        glfwPollEvents();
    }

    // Clean up OpenCL resources before destroying the window and glfw.
    context = nullptr;
    commandQueue = nullptr;
    program = nullptr;
    kernelJulia = nullptr;
    kernelSobel = nullptr;
    sampler = nullptr;
    memTmp = nullptr;
    memDst = nullptr;

    glfwDestroyWindow(pWindow);
    glfwTerminate();

    return 0;
}
