#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return 1;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl << std::endl;

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;

        std::cout << "Number of SMs: " << deviceProp.multiProcessorCount << std::endl;

        std::cout << "Total global memory: " << deviceProp.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;

        int clockRate;
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev);
        std::cout << "GPU Clock rate: " << clockRate / 1000 << " MHz" << std::endl;

        int memClockRate;
        cudaDeviceGetAttribute(&memClockRate, cudaDevAttrMemoryClockRate, dev);
        std::cout << "Memory Clock rate: " << memClockRate / 1000 << " MHz" << std::endl;

        int memBusWidth;
        cudaDeviceGetAttribute(&memBusWidth, cudaDevAttrGlobalMemoryBusWidth, dev);
        std::cout << "Memory Bus Width: " << memBusWidth << " bits" << std::endl;

        float memBandwidth = 2.0f * static_cast<float>(memClockRate) * (memBusWidth / 8.0) / 1.0e6f;
        std::cout << "Memory Bandwidth: " << memBandwidth << " GB/s" << std::endl;

        std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max threads per SM: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Warp size: " << deviceProp.warpSize << std::endl;

        std::cout << std::endl;
    }

    return 0;
}