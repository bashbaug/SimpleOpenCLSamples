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

bool animate = false;
bool redraw = false;
bool vsync = true;

size_t width = 1024;
size_t height = 1024;

size_t numBodies = 1024;
size_t groupSize = 0;

size_t currentPos = 0;

cl::Context context;
cl::CommandQueue commandQueue;
cl::Kernel kernel;
cl::Buffer pos[2];
cl::Buffer vel;

static const char kernelString[] = R"CLC(
__kernel void nbody_step(
    const __global float4* pos,
    __global float4* nextPos,
    __global float4* vel)
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

    float4 newPos = 0;
    newPos.xyz = myPos + myVel * deltaTime;
    newPos.w = myMass;

    float4 newVel = 0;
    newVel.xyz = myVel + myAcc * deltaTime;
    newVel *= dampen;

    nextPos[ get_global_id(0) ] = newPos;
    vel[ get_global_id(0) ] = newVel;
}
)CLC";

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

    pos[0] = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, numBodies * sizeof(cl_float4), init_pos.data());
    pos[1] = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, numBodies * sizeof(cl_float4));

    vel = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, numBodies * sizeof(cl_float4), init_vel.data());
}

static void resize(GLFWwindow* pWindow, int width, int height)
{
    glViewport(0, 0, width, height);
}

static void display(void)
{
    size_t nextPos = 1 - currentPos;

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

    kernel.setArg(0, pos[currentPos]);
    kernel.setArg(1, pos[nextPos]);
    kernel.setArg(2, vel);

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

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);

    glColor3f(1.0f, 0.6f, 0.0f);

    const cl_float4* p = (const cl_float4*)commandQueue.enqueueMapBuffer(pos[currentPos], CL_TRUE, CL_MAP_READ, 0, numBodies * sizeof(cl_float4));

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, sizeof(cl_float4), p);
    glDrawArrays(GL_POINTS, 0, (GLsizei)numBodies);
    glDisableClientState(GL_VERTEX_ARRAY);

    commandQueue.enqueueUnmapMemObject(pos[currentPos], (void*)p);

    GLenum  gl_error = glGetError();
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "Error: OpenGL generated %x!\n", gl_error);
    }

    currentPos = nextPos;
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
        bool paused = false;

        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("n", "numbodies", "Number of Bodies", numBodies, &numBodies);
        op.add<popl::Value<size_t>>("g", "groupsize", "Group Size", groupSize, &groupSize);
        op.add<popl::Value<size_t>>("w", "width", "Render Width", width, &width);
        op.add<popl::Value<size_t>>("h", "height", "Render Height", height, &height);
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
                "Usage: nbodygl [options]\n"
                "%s", op.help().c_str());
            return -1;
        }

        animate = !paused;
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

    context = cl::Context(devices[deviceIndex]);
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

    // Clean up OpenCL resources before destroying the window and glfw.
    context = nullptr;
    commandQueue = nullptr;
    program = nullptr;
    kernel = nullptr;
    pos[0] = nullptr;
    pos[1] = nullptr;
    vel = nullptr;

    glfwDestroyWindow(pWindow);
    glfwTerminate();

    return 0;
}
