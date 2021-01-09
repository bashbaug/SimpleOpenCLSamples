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

#include <CL/opencl.hpp>

#include "nanoglut.h"

#include <chrono>
#include <ctime>

#if !defined(GL_CLAMP_TO_EDGE)
#define GL_CLAMP_TO_EDGE                  0x812F
#endif

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

static void resize(int width, int height)
{
    glViewport(0, 0, width, height);
}

static void display(void)
{
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

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glEnable( GL_TEXTURE_2D );

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

    commandQueue.enqueueUnmapMemObject(
        mem,
        pixels);
}

static void keyboard(unsigned char key, int x, int y)
{
    switch( key )
    {
    case 27:
        nglutLeaveMainLoop();
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

    cl::Context context{devices[deviceIndex]};
    commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
    kernel = cl::Kernel{ program, "Julia" };

    mem = cl::Image2D{
        context,
        CL_MEM_WRITE_ONLY,
        cl::ImageFormat{CL_RGBA, CL_FLOAT},
        gwx, gwy };

    nglutReshapeFunc(resize);
    nglutDisplayFunc(display);
    nglutKeyboardFunc(keyboard);

    nglutMainLoop();

    nglutShutdown();

    return 0;
}