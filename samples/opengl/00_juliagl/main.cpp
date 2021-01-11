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

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/opencl.hpp>

#include "nanoglut.h"
#include "util.hpp"

#include <chrono>
#include <math.h>

#if !defined(GL_CLAMP_TO_EDGE)
#define GL_CLAMP_TO_EDGE                  0x812F
#endif

bool use_cl_khr_gl_sharing = true;
bool use_cl_khr_gl_event = true;
bool animate = false;

size_t gwx = 512;
size_t gwy = 512;
size_t lwx = 0;
size_t lwy = 0;

float cr = -0.123f;
float ci =  0.745f;

cl::CommandQueue commandQueue;
cl::Kernel kernel;
cl::Image2D mem;

static const char kernelString[] = R"CLC(
kernel void Julia( write_only image2d_t dst, float cr, float ci )
{
    const float cMinX = -1.5f;
    const float cMaxX =  1.5f;
    const float cMinY = -1.5f;
    const float cMaxY =  1.5f;

    const int cWidth = get_global_size(0);
    const int cIterations = 16;

    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);

    float a = x * ( cMaxX - cMinX ) / cWidth + cMinX;
    float b = y * ( cMaxY - cMinY ) / cWidth + cMinY;

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
)CLC";

cl::Context createContext(const cl::Platform& platform, const cl::Device& device)
{
    const cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform(),
#if defined(WIN32)
        CL_GL_CONTEXT_KHR, (cl_context_properties)nglut_hRC,
        CL_WGL_HDC_KHR, (cl_context_properties)nglut_hDC,
#elif defined(__linux__)
        CL_GL_CONTEXT_KHR, (cl_context_properties)nglut_hRC,
        CL_GLX_DISPLAY_KHR, (cl_context_properties)nglut_hDisplay,
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
                    printf("  %s\n", cl::Device(check).getInfo<CL_DEVICE_NAME>().c_str());
                }
            }
        }

        return cl::Context(device, props);
    }

    return cl::Context(device);
}

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
        // Note: this is an extension API, but it is exported directly from
        // the ICD loader.
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

static void resize(int width, int height)
{
    glViewport(0, 0, width, height);
}

static void display(void)
{
    if (use_cl_khr_gl_sharing) {
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

    kernel.setArg(0, mem);
    kernel.setArg(1, cr);
    kernel.setArg(2, ci);

    cl::NDRange lws;    // NullRange by default.

    if( lwx > 0 && lwy > 0 )
    {
        lws = cl::NDRange{lwx, lwy};
    }

    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{gwx, gwy},
        lws);

    if (use_cl_khr_gl_sharing) {
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

    nglutSwapBuffers();
}

static void keyboard(unsigned char key, int x, int y)
{
    switch( key )
    {
    case 27:
        nglutLeaveMainLoop();
        break;
    case ' ':
        animate = !animate;
        break;

    case 'A':
        cr += 0.005f;
        break;
    case 'Z':
        cr -= 0.005f;
        break;

    case 'S':
        ci += 0.005f;
        break;
    case 'X':
        ci -= 0.005f;
        break;
    } 

    nglutPostRedisplay();
}

static void idle(void)
{
    if (animate) {
        static size_t frame = 0;
        static std::chrono::system_clock::time_point start = 
            std::chrono::system_clock::now();

        float fcr = (frame % 599) / 599.f * 2.0f * CL_M_PI_F;
        float fci = (frame % 773) / 773.f * 2.0f * CL_M_PI_F;
        cr = sinf(fcr);
        ci = sinf(fci);

        if (++frame % 100 == 0) {
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            printf("FPS: %.1f\n", 100 / elapsed_seconds.count());
            start = end;
        }

        nglutPostRedisplay();
    }
}

int main(
    int argc,
    char** argv )
{
    bool printUsage = false;
    int platformIndex = 0;
    int deviceIndex = 0;

    if( argc < 1 )
    {
        printUsage = true;
    }
    else
    {
        for( size_t i = 1; i < argc; i++ )
        {
            if( !strcmp( argv[i], "-d" ) )
            {
                if( ++i < argc )
                {
                    deviceIndex = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-p" ) )
            {
                if( ++i < argc )
                {
                    platformIndex = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-hostcopy" ) )
            {
                use_cl_khr_gl_sharing = false;
            }
            else if( !strcmp( argv[i], "-hostsync" ) )
            {
                use_cl_khr_gl_event = false;
            }
            else if( !strcmp( argv[i], "-gwx" ) )
            {
                if( ++i < argc )
                {
                    gwx = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-gwy" ) )
            {
                if( ++i < argc )
                {
                    gwy = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-lwx" ) )
            {
                if( ++i < argc )
                {
                    lwx = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-lwy" ) )
            {
                if( ++i < argc )
                {
                    lwy = strtol(argv[i], NULL, 10);
                }
            }
            else
            {
                printUsage = true;
            }
        }
    }
    if( printUsage )
    {
        fprintf(stderr,
            "Usage: juliagl   [options]\n"
            "Options:\n"
            "      -d: Device Index (default = 0)\n"
            "      -p: Platform Index (default = 0)\n"
            "      -hostcopy: Do not use cl_khr_gl_sharing\n"
            "      -hostsync: Do not use cl_khr_gl_event\n"
            "      -gwx: Global Work Size X AKA Image Width (default = 512)\n"
            "      -gwy: Global Work Size Y AKA Image Height (default = 512)\n"
            "      -lwx: Local Work Size X (default = 0 = NULL Local Work Size)\n"
            "      -lwy: Local Work Size Y (default = 0 = Null Local Work size)\n"
            );

        return -1;
    }

    //glutInit(&argc, argv);
    nglutInitWindowSize((int)gwx, (int)gwy);
    //glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    nglutCreateWindow("Julia Set with OpenGL");

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

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
    kernel = cl::Kernel{ program, "Julia" };

    mem = createImage(context);

    nglutReshapeFunc(resize);
    nglutDisplayFunc(display);
    nglutKeyboardFunc(keyboard);
    nglutIdleFunc(idle);

    nglutMainLoop();

    nglutShutdown();

    return 0;
}
