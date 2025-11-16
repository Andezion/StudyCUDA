#include <iostream>

__global__ void hello_world()
{
    printf("Hello bitch, %d, %d, %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
}

__global__ void hello_world_2()
{
    printf("Hello world, %d, %d, %d\n", blockDim.x, blockDim.y, blockDim.z);
}

int main()
{
    cudaDeviceProp deviceProp{};

    cudaGetDeviceProperties(&deviceProp, 0);

    printf("Device name : %s\n", deviceProp.name);

    printf("Total global memory : %d MB\n", deviceProp.totalGlobalMem / 1024 / 1024);
    printf("Shared memory per block : %d\n", deviceProp.sharedMemPerBlock);
    printf("Registers per block : %d\n", deviceProp.regsPerBlock);

    printf("Warp size : %d\n", deviceProp.warpSize);
    printf("Memory pitch : %d\n", deviceProp.memPitch);
    printf("Max threads per block : %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max threads dimensions : x = %d, y = %d, z = %d\n",
                        deviceProp.maxThreadsDim[0],
                        deviceProp.maxThreadsDim[1],
                        deviceProp.maxThreadsDim[2]);

    printf("Max grid size: x = %d, y = %d, z = %d\n",
                        deviceProp.maxGridSize[0],
                        deviceProp.maxGridSize[1],
                        deviceProp.maxGridSize[2]);

    printf("Total constant memory: %d\n", deviceProp.totalConstMem);
    printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Texture alignment: %d\n", deviceProp.textureAlignment);
    //printf("Device overlap: %d\n", deviceProp.deviceOverlap);
    printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
    //printf("Kernel execution timeout enabled: %s\n", deviceProp.kernelExecTimeoutEnabled ? "true" : "false");

    scanf("");
    return 0;
}
