#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at %s:%d\n", \
                   cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void measureBandwidth(const size_t dataSize)
{
    float *h_data, *d_data;
    cudaEvent_t start, stop;
    float elapsedTime;

    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_data), dataSize));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_data), dataSize));

    for (size_t i = 0; i < dataSize / sizeof(float); i++)
    {
        h_data[i] = static_cast<float>(i);
    }

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, nullptr));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop, nullptr));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    const float bandwidthHostToDevice = (dataSize / (1024.0 * 1024.0 * 1024.0)) / (elapsedTime / 1000.0);
    printf("CPU -> GPU: %.2f Gb/sec (time: %.3f ms)\n",
           bandwidthHostToDevice, elapsedTime);

    CUDA_CHECK(cudaEventRecord(start, nullptr));
    CUDA_CHECK(cudaMemcpy(h_data, d_data, dataSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop, nullptr));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    float bandwidthDeviceToHost = (dataSize / (1024.0 * 1024.0 * 1024.0)) / (elapsedTime / 1000.0);
    printf("GPU -> CPU: %.2f Gb/sec (time: %.3f ms)\n",
           bandwidthDeviceToHost, elapsedTime);

    CUDA_CHECK(cudaFreeHost(h_data));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    printf("Comparing CPU and GPU\n\n");

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);

    int memClockRate;
    cudaDeviceGetAttribute(&memClockRate, cudaDevAttrMemoryClockRate, 0);
    printf("Theoretical max rate: %.2f Gb/sec\n\n",
           2.0 * memClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

    size_t sizes[] = {
        1 * 1024 * 1024,      // 1 МБ
        10 * 1024 * 1024,     // 10 МБ
        100 * 1024 * 1024,    // 100 МБ
        500 * 1024 * 1024     // 500 МБ
    };

    for (unsigned long long size : sizes)
    {
        printf("Size of data: %.2f Mb\n", size / (1024.0 * 1024.0));
        measureBandwidth(size);
        printf("\n");
    }

    return 0;
}