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

// The code in this sample was derived from several samples in the Vulkan
// Tutorial: https://vulkan-tutorial.com
//
// The code samples in the Vulkan Tutorial are licensed as CC0 1.0 Universal.

#include <popl/popl.hpp>

#include <CL/opencl.hpp>
#if !defined(cl_khr_external_memory)
#error cl_khr_external_memory not found, please update your OpenCL headers!
#endif
#if !defined(cl_khr_external_semaphore)
#error cl_khr_external_semaphore not found, please update your OpenCL headers!
#endif

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "util.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include <math.h>

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

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamilyIndices {
    uint32_t graphicsFamily;
    uint32_t presentFamily;

    QueueFamilyIndices() :
        graphicsFamily(~0),
        presentFamily(~0) {}

    bool isComplete() {
        return graphicsFamily != ~0 && presentFamily != ~0;
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class NBodyVKApplication {
public:
    void run(int argc, char** argv) {
        commandLine(argc, argv);
        initWindow();
        initOpenCL();
        initVulkan();
        initOpenCLMems();
        initOpenCLSemaphores();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;

    bool animate = false;
    bool redraw = false;

    uint32_t lastImage = 0;

    size_t width = 1024;
    size_t height = 1024;

    size_t numBodies = 1024;
    size_t groupSize = 0;

    bool vsync = true;
    size_t startFrame = 0;
    size_t frame = 0;
    std::chrono::system_clock::time_point start =
        std::chrono::system_clock::now();

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkCommandPool commandPool;

    bool deviceLocalBuffers = true;
    std::vector<VkBuffer> vertexBuffers;
    std::vector<VkDeviceMemory> vertexBufferMemories;

    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkSemaphore> openclFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;
    size_t currentFrame = 0;

#ifdef _WIN32
    PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR = NULL;
    PFN_vkGetSemaphoreWin32HandleKHR vkGetSemaphoreWin32HandleKHR = NULL;
#elif defined(__linux__)
    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = NULL;
    PFN_vkGetSemaphoreFdKHR vkGetSemaphoreFdKHR = NULL;
#endif

    int platformIndex = 0;
    int deviceIndex = 0;

    bool useExternalMemory = true;
    bool useExternalSemaphore = true;

    cl_external_memory_handle_type_khr externalMemType = 0;
    cl::Context context;
    cl::CommandQueue commandQueue;
    cl::Kernel kernel;

    std::vector<cl::Buffer> pos;
    cl::Buffer vel;
    std::vector<cl_semaphore_khr> signalSemaphores;

    clEnqueueAcquireExternalMemObjectsKHR_fn clEnqueueAcquireExternalMemObjectsKHR = NULL;
    clEnqueueReleaseExternalMemObjectsKHR_fn clEnqueueReleaseExternalMemObjectsKHR = NULL;

    clCreateSemaphoreWithPropertiesKHR_fn clCreateSemaphoreWithPropertiesKHR = NULL;
    clEnqueueSignalSemaphoresKHR_fn clEnqueueSignalSemaphoresKHR = NULL;
    clReleaseSemaphoreKHR_fn clReleaseSemaphoreKHR = NULL;

    void commandLine(int argc, char** argv) {
        bool hostCopy = false;
        bool hostSync = false;
        bool noDeviceLocal = false;
        bool immediate = false;

        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Switch>("", "hostcopy", "Do not use cl_khr_external_memory", &hostCopy);
        op.add<popl::Switch>("", "hostsync", "Do not use cl_khr_external_semaphore", &hostSync);
        op.add<popl::Switch>("", "nodevicelocal", "Do not use device local buffers", &noDeviceLocal);
        op.add<popl::Value<size_t>>("n", "numbodies", "Number of Bodies", numBodies, &numBodies);
        op.add<popl::Value<size_t>>("g", "groupsize", "Group Size", groupSize, &groupSize);
        op.add<popl::Value<size_t>>("w", "width", "Render Width", width, &width);
        op.add<popl::Value<size_t>>("h", "height", "Render Height", height, &height);
        op.add<popl::Switch>("", "immediate", "Prefer VK_PRESENT_MODE_IMMEDIATE_KHR (no vsync)", &immediate);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: nbodyvk [options]\n"
                "%s", op.help().c_str());
            throw std::runtime_error("exiting.");
        }

        deviceLocalBuffers = !noDeviceLocal;
        useExternalMemory = !hostCopy;
        useExternalSemaphore = !hostSync;
        vsync = !immediate;
    }

    void initWindow() {
        if (!glfwInit()) {
            throw std::runtime_error("failed to initialize glfw!");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow((int)width, (int)height, "N-Body Simulation with Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
    }

    void initOpenCL() {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        printf("Running on platform: %s\n",
            platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

        std::vector<cl::Device> devices;
        platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        printf("Running on device: %s\n",
            devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

        checkOpenCLExternalMemorySupport(devices[deviceIndex]);
        checkOpenCLExternalSemaphoreSupport(devices[deviceIndex]);

        if (useExternalMemory) {
            clEnqueueAcquireExternalMemObjectsKHR = (clEnqueueAcquireExternalMemObjectsKHR_fn)
                clGetExtensionFunctionAddressForPlatform( platforms[platformIndex](), "clEnqueueAcquireExternalMemObjectsKHR");
            clEnqueueReleaseExternalMemObjectsKHR = (clEnqueueReleaseExternalMemObjectsKHR_fn)
                clGetExtensionFunctionAddressForPlatform( platforms[platformIndex](), "clEnqueueReleaseExternalMemObjectsKHR");
            if (clEnqueueAcquireExternalMemObjectsKHR == NULL ||
                clEnqueueReleaseExternalMemObjectsKHR == NULL) {
                throw std::runtime_error("couldn't get function pointers for cl_khr_external_memory");
            }
        }

        if (useExternalSemaphore) {
            clCreateSemaphoreWithPropertiesKHR = (clCreateSemaphoreWithPropertiesKHR_fn)
                clGetExtensionFunctionAddressForPlatform(platforms[platformIndex](), "clCreateSemaphoreWithPropertiesKHR");
            clEnqueueSignalSemaphoresKHR = (clEnqueueSignalSemaphoresKHR_fn)
                clGetExtensionFunctionAddressForPlatform(platforms[platformIndex](), "clEnqueueSignalSemaphoresKHR");
            clReleaseSemaphoreKHR = (clReleaseSemaphoreKHR_fn)
                clGetExtensionFunctionAddressForPlatform(platforms[platformIndex](), "clReleaseSemaphoreKHR");
            if (clCreateSemaphoreWithPropertiesKHR == NULL ||
                clEnqueueSignalSemaphoresKHR == NULL ||
                clReleaseSemaphoreKHR == NULL) {
                throw std::runtime_error("couldn't get function pointers for cl_khr_external_semaphore");
            }
        }

        context = cl::Context{devices[deviceIndex]};
        commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

        cl::Program program{ context, kernelString };
        program.build();
        kernel = cl::Kernel{ program, "nbody_step" };
    }

    void initOpenCLMems() {
        std::mt19937 gen;
        std::uniform_real_distribution<float> rand_pos(-0.01f, 0.01f);
        std::uniform_real_distribution<float> rand_mass(0.1f, 1.0f);

        std::vector<cl_float4> init_pos(numBodies);
        for (size_t i = 0; i < numBodies; i++) {
            // X, Y, and Z position:
            init_pos[i].s[0] = rand_pos(gen);
            init_pos[i].s[1] = rand_pos(gen);
            init_pos[i].s[2] = rand_pos(gen);

            // Mass:
            init_pos[i].s[3] = rand_mass(gen);
        }

        pos.resize(swapChainImages.size());
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            if (useExternalMemory) {
#ifdef _WIN32
                HANDLE handle = NULL;
                VkMemoryGetWin32HandleInfoKHR getWin32HandleInfo{};
                getWin32HandleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
                getWin32HandleInfo.memory = vertexBufferMemories[i];
                getWin32HandleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
                vkGetMemoryWin32HandleKHR(device, &getWin32HandleInfo, &handle);

                const cl_mem_properties props[] = {
                    externalMemType,
                    (cl_mem_properties)handle,
                    0,
                };
#elif defined(__linux__)
                int fd = 0;
                VkMemoryGetFdInfoKHR getFdInfo{};
                getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
                getFdInfo.memory = vertexBufferMemories[i];
                getFdInfo.handleType =
                    externalMemType == CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR ?
                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT :
                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
                vkGetMemoryFdKHR(device, &getFdInfo, &fd);

                const cl_mem_properties props[] = {
                    externalMemType,
                    (cl_mem_properties)fd,
                    0,
                };
#else
                const cl_mem_properties* props = NULL;
#endif
                pos[i] = cl::Buffer{
                    clCreateBufferWithProperties(
                        context(),
                        props,
                        CL_MEM_READ_WRITE,
                        sizeof(cl_float4) * numBodies,
                        NULL,
                        NULL)};
            } else {
                pos[i] = cl::Buffer{context, CL_MEM_READ_WRITE, sizeof(cl_float4) * numBodies};
            }

            commandQueue.enqueueWriteBuffer(pos[i], CL_TRUE, 0, sizeof(cl_float4) * numBodies, init_pos.data());
        }

        vel = cl::Buffer{context, CL_MEM_READ_WRITE, sizeof(cl_float4) * numBodies};
        commandQueue.enqueueFillBuffer(vel, 0.0f, 0, sizeof(cl_float4) * numBodies);
        commandQueue.finish();
    }

    void initOpenCLSemaphores() {
        if (useExternalSemaphore) {
            signalSemaphores.resize(MAX_FRAMES_IN_FLIGHT);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createOpenCLSemaphoreFromVulkanSemaphore(openclFinishedSemaphores[i], signalSemaphores[i]);
            }
        }
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createVertexBuffers();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop() {
        glfwSetKeyCallback(window, keyboard);

        while (!glfwWindowShouldClose(window)) {
            if (animate || redraw) {
                drawFrame();
            }
            glfwPollEvents();
        }

        vkDeviceWaitIdle(device);
    }

    void keyboard(int key, int scancode, int action, int mods) {
        if (action == GLFW_PRESS || action == GLFW_REPEAT) {
            redraw = true;

            switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_SPACE:
                animate = !animate;
                printf("animation is %s\n", animate ? "ON" : "OFF");
                break;

            case GLFW_KEY_S:
                printf("stepping...\n");
                redraw = true;
                break;
            }
        }
    }

    void cleanup() {
        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        // vkFreeCommandBuffers?

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);

        for (auto buffer : vertexBuffers) {
            vkDestroyBuffer(device, buffer, nullptr);
        }
        for (auto bufferMemory : vertexBufferMemories) {
            vkFreeMemory(device, bufferMemory, nullptr);
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            if (useExternalSemaphore) {
                vkDestroySemaphore(device, openclFinishedSemaphores[i], nullptr);
            }
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "NBody OpenCL+Vulkan Sample";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        if (useExternalMemory || useExternalSemaphore) {
            appInfo.apiVersion = VK_API_VERSION_1_1;
        } else {
            appInfo.apiVersion = VK_API_VERSION_1_0;
        }

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;

            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }

#ifdef _WIN32
        if (useExternalMemory) {
            vkGetMemoryWin32HandleKHR = (PFN_vkGetMemoryWin32HandleKHR)vkGetInstanceProcAddr(instance, "vkGetMemoryWin32HandleKHR");
            if (vkGetMemoryWin32HandleKHR == NULL) {
                throw std::runtime_error("couldn't get function pointer for vkGetMemoryWin32HandleKHR");
            }
        }
        if (useExternalSemaphore) {
            vkGetSemaphoreWin32HandleKHR = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetInstanceProcAddr(instance, "vkGetSemaphoreWin32HandleKHR");
            if (vkGetSemaphoreWin32HandleKHR == NULL) {
                throw std::runtime_error("couldn't get function pointer for vkGetSemaphoreWin32HandleKHR");
            }
        }
#elif defined(__linux__)
        if (useExternalMemory) {
            vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetInstanceProcAddr(instance, "vkGetMemoryFdKHR");
            if (vkGetMemoryFdKHR == NULL) {
                throw std::runtime_error("couldn't get function pointer for vkGetMemoryFdKHR");
            }
        }
        if (useExternalSemaphore) {
            vkGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetInstanceProcAddr(instance, "vkGetSemaphoreFdKHR");
            if (vkGetSemaphoreFdKHR == NULL) {
                throw std::runtime_error("couldn't get function pointer for vkGetSemaphoreFdKHR");
            }
        }
#endif
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily, indices.presentFamily};

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        auto extensions = getRequiredDeviceExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily, 0, &presentQueue);
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily, indices.presentFamily};

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createGraphicsPipeline() {
        auto vertShaderCode = readFile("nbodyvk.vert.spv");
        auto fragShaderCode = readFile("nbodyvk.frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        VkVertexInputBindingDescription vertexInputBindingDescription{};
        vertexInputBindingDescription.binding = 0;
        vertexInputBindingDescription.stride = sizeof(cl_float4);
        vertexInputBindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription vertexInputAttributeDescription{};
        vertexInputAttributeDescription.binding = 0;
        vertexInputAttributeDescription.location = 0;
        vertexInputAttributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexInputAttributeDescription.offset = 0;

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &vertexInputBindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = &vertexInputAttributeDescription;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) swapChainExtent.width;
        viewport.height = (float) swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    void createVertexBuffers() {
        VkMemoryPropertyFlags properties =
            deviceLocalBuffers && useExternalMemory ?
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT :
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        vertexBuffers.resize(swapChainImages.size());
        vertexBufferMemories.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            createShareableBuffer(
                sizeof(cl_float4) * numBodies,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                properties,
                vertexBuffers[i],
                vertexBufferMemories[i]);
        }
    }

    void createShareableBuffer(size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
    {
        VkExternalMemoryBufferCreateInfo externalMemCreateInfo{};
        externalMemCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;

#ifdef _WIN32
        externalMemCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#elif defined(__linux__)
        externalMemCreateInfo.handleTypes =
            externalMemType == CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR ?
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT :
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
#endif

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        if (useExternalMemory) {
            bufferInfo.pNext = &externalMemCreateInfo;
        }
        bufferInfo.size = static_cast<VkDeviceSize>(size);
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create vertex buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkExportMemoryAllocateInfo exportMemoryAllocInfo{};
        exportMemoryAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        exportMemoryAllocInfo.handleTypes = externalMemCreateInfo.handleTypes;

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        if (useExternalMemory) {
            allocInfo.pNext = &exportMemoryAllocInfo;
        }
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate vertex buffer memory!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    void createOpenCLSemaphoreFromVulkanSemaphore(VkSemaphore srcSemaphore, cl_semaphore_khr& semaphore) {
#ifdef _WIN32
        HANDLE handle = NULL;
        VkSemaphoreGetWin32HandleInfoKHR getWin32HandleInfo{};
        getWin32HandleInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
        getWin32HandleInfo.semaphore = srcSemaphore;
        getWin32HandleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        vkGetSemaphoreWin32HandleKHR(device, &getWin32HandleInfo, &handle);

        const cl_semaphore_properties_khr props[] = {
            CL_SEMAPHORE_TYPE_KHR,
            CL_SEMAPHORE_TYPE_BINARY_KHR,
            CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR,
            (cl_semaphore_properties_khr)handle,
            0,
        };
#elif defined(__linux__)
        int fd = 0;
        VkSemaphoreGetFdInfoKHR getFdInfo{};
        getFdInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
        getFdInfo.semaphore = srcSemaphore;
        getFdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        vkGetSemaphoreFdKHR(device, &getFdInfo, &fd);

        const cl_semaphore_properties_khr props[] = {
            CL_SEMAPHORE_TYPE_KHR,
            CL_SEMAPHORE_TYPE_BINARY_KHR,
            CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR,
            (cl_semaphore_properties_khr)fd,
            0,
        };
#else
        const cl_mem_properties* props = NULL;
#endif

        semaphore = clCreateSemaphoreWithPropertiesKHR(
            context(),
            props,
            NULL);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void createCommandBuffers() {
        commandBuffers.resize(swapChainFramebuffers.size());

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }

        for (size_t i = 0; i < commandBuffers.size(); i++) {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("failed to begin recording command buffer!");
            }

            VkRenderPassBeginInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = renderPass;
            renderPassInfo.framebuffer = swapChainFramebuffers[i];
            renderPassInfo.renderArea.offset = {0, 0};
            renderPassInfo.renderArea.extent = swapChainExtent;

            VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
            renderPassInfo.clearValueCount = 1;
            renderPassInfo.pClearValues = &clearColor;

            vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

                vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

                VkDeviceSize offsets[] = {0};
                vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, &vertexBuffers[i], offsets);

                vkCmdDraw(commandBuffers[i], static_cast<uint32_t>(numBodies), 1, 0, 0);

            vkCmdEndRenderPass(commandBuffers[i]);

            if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer!");
            }
        }
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

        VkExportSemaphoreCreateInfo exportSemaphoreCreateInfo{};
        exportSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;

#ifdef _WIN32
        exportSemaphoreCreateInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#elif defined(__linux__)
        exportSemaphoreCreateInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }

        if (useExternalSemaphore) {
            openclFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);

            semaphoreInfo.pNext = &exportSemaphoreCreateInfo;

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &openclFinishedSemaphores[i]) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create synchronization objects for interop!");
                }
            }
        }
    }

    void updateVertexBuffer(uint32_t lastImage, uint32_t currentImage) {
        if (useExternalMemory) {
            cl_mem acquireMems[2] = {
                pos[lastImage](),
                pos[currentImage](),
            };
            clEnqueueAcquireExternalMemObjectsKHR(
                commandQueue(),
                2,
                acquireMems,
                0,
                NULL,
                NULL);
        }

        if (lastImage != currentImage) {
            kernel.setArg(0, pos[lastImage]);
            kernel.setArg(1, pos[currentImage]);
            kernel.setArg(2, vel);

            cl::NDRange lws;    // NullRange by default.
            if (groupSize > 0) {
                lws = cl::NDRange{groupSize};
            }

            commandQueue.enqueueNDRangeKernel(
                kernel,
                cl::NullRange,
                cl::NDRange{numBodies},
                lws);
        }

        if (useExternalMemory) {
            cl_mem releaseMems[2] = {
                pos[lastImage](),
                pos[currentImage](),
            };
            clEnqueueReleaseExternalMemObjectsKHR(
                commandQueue(),
                2,
                releaseMems,
                0,
                NULL,
                NULL);
            if (useExternalSemaphore) {
                clEnqueueSignalSemaphoresKHR(
                    commandQueue(),
                    1,
                    &signalSemaphores[currentFrame],
                    nullptr,
                    0,
                    nullptr,
                    nullptr);
                commandQueue.flush();
            } else {
                commandQueue.finish();
            }
        } else {
            void* srcData = commandQueue.enqueueMapBuffer(
                pos[currentImage],
                CL_TRUE,
                CL_MAP_READ,
                0,
                sizeof(cl_float4) * numBodies);

            void* dstData;
            vkMapMemory(device, vertexBufferMemories[currentImage], 0, sizeof(cl_float4) * numBodies, 0, &dstData);
                memcpy(dstData, srcData, sizeof(cl_float4) * numBodies);
            vkUnmapMemory(device, vertexBufferMemories[currentImage]);

            commandQueue.enqueueUnmapMemObject(pos[currentImage], srcData);
            commandQueue.flush();
        }
    }

    void drawFrame() {
        if (animate) {
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

        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        updateVertexBuffer(lastImage, imageIndex);

        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
            vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        }
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        std::vector<VkSemaphore> waitSemaphores;
        std::vector<VkPipelineStageFlags> waitStages;
        waitSemaphores.push_back(imageAvailableSemaphores[currentFrame]);
        waitStages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        if (useExternalMemory && useExternalSemaphore) {
            waitSemaphores.push_back(openclFinishedSemaphores[currentFrame]);
            waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }
        submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
        submitInfo.pWaitSemaphores = waitSemaphores.data();
        submitInfo.pWaitDstStageMask = waitStages.data();

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;

        presentInfo.pImageIndices = &imageIndex;

        vkQueuePresentKHR(presentQueue, &presentInfo);

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        lastImage = imageIndex;
    }

    void checkOpenCLExternalMemorySupport(cl::Device& device) {
        if (checkDeviceForExtension(device, "cl_khr_external_memory")) {
            printf("Device supports cl_khr_external_memory.\n");
            printf("Supported external memory handle types:\n");
            std::vector<cl_external_memory_handle_type_khr> types =
                device.getInfo<CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR>();
            for (auto type : types) {
                #define CASE_TO_STRING(_e) case _e: printf("\t%s\n", #_e); break;
                switch(type) {
                CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR);
                CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR);
                CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR);
                CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_D3D11_TEXTURE_KHR);
                CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_D3D11_TEXTURE_KMT_KHR);
                CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_D3D12_HEAP_KHR);
                CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_D3D12_RESOURCE_KHR);
                CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR);
                default: printf("Unknown cl_external_memory_handle_type_khr %04X\n", type);
                }
                #undef CASE_TO_STRING
            }
#ifdef _WIN32
            if (std::find(types.begin(), types.end(), CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR) != types.end()) {
                externalMemType = CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR;
            } else {
                printf("Couldn't find a compatible external memory type (sample supports OPAQUE_WIN32).\n");
                useExternalMemory = false;
            }
#elif defined(__linux__)
            if (std::find(types.begin(), types.end(), CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR) != types.end()) {
                externalMemType = CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR;
            } else if (std::find(types.begin(), types.end(), CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR) != types.end()) {
                externalMemType = CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR;
            } else {
                printf("Couldn't find a compatible external memory type (sample supports DMA_BUF or OPAQUE_FD).\n");
                useExternalMemory = false;
            }
#endif
        } else {
            printf("Device does not support cl_khr_external_memory.\n");
            useExternalMemory = false;
        }
    }

    void checkOpenCLExternalSemaphoreSupport(cl::Device& device) {
        if (checkDeviceForExtension(device, "cl_khr_external_semaphore")) {
            printf("Device supports cl_khr_external_semaphore.\n");
            printf("Supported external semaphore import handle types:\n");
            std::vector<cl_external_semaphore_handle_type_khr> types =
                device.getInfo<CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR>();
            for (auto type : types) {
                #define CASE_TO_STRING(_e) case _e: printf("\t%s\n", #_e); break;
                switch(type) {
                CASE_TO_STRING(CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR);
                CASE_TO_STRING(CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR);
                CASE_TO_STRING(CL_SEMAPHORE_HANDLE_SYNC_FD_KHR);
                CASE_TO_STRING(CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR);
                CASE_TO_STRING(CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR);
                default: printf("Unknown cl_external_semaphore_handle_type_khr %04X\n", type);
                }
                #undef CASE_TO_STRING
            }
#ifdef _WIN32
            if (std::find(types.begin(), types.end(), CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR) == types.end()) {
                printf("Couldn't find a compatible external semaphore type (sample supports OPAQUE_WIN32).\n");
                useExternalSemaphore = false;
            }
#elif defined(__linux__)
            if (std::find(types.begin(), types.end(), CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR) == types.end()) {
                printf("Couldn't find a compatible external semaphore type (sample supports OPAQUE_FD).\n");
                useExternalSemaphore = false;
            }
#endif
        } else {
            printf("Device does not support cl_khr_external_semaphore.\n");
            useExternalSemaphore = false;
        }
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (vsync) {
                if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                    return availablePresentMode;
                }
            } else {
                if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
                    return availablePresentMode;
                }
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != UINT32_MAX) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(actualExtent.width, capabilities.maxImageExtent.width));
            actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(actualExtent.height, capabilities.maxImageExtent.height));

            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        auto extensions = getRequiredDeviceExtensions();
        std::set<std::string> requiredExtensions(extensions.begin(), extensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (useExternalMemory || useExternalSemaphore) {
            extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        }
        if (useExternalMemory) {
            extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
        }
        if (useExternalMemory && externalMemType == CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR) {
            extensions.push_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
        }
        if (useExternalSemaphore) {
            extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
        }
        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    std::vector<const char*> getRequiredDeviceExtensions() {
        std::vector<const char*> extensions(deviceExtensions);

        if (useExternalMemory) {
            extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
#ifdef _WIN32
            extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#elif defined(__linux__)
            extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
        }
        if (useExternalSemaphore) {
            extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#ifdef _WIN32
            extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#elif defined(__linux__)
            extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif
        }

        return extensions;
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        fprintf(stderr, "validation layer: %s\n", pCallbackData->pMessage);

        return VK_FALSE;
    }

    static void keyboard(GLFWwindow* pWindow, int key, int scancode, int action, int mods)
    {
        auto pApp = (NBodyVKApplication*)glfwGetWindowUserPointer(pWindow);
        pApp->keyboard(key, scancode, action, mods);
    }
};

int main(int argc, char** argv) {
    NBodyVKApplication app;

    try {
        app.run(argc, argv);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
