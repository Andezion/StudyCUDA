#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

double M_PI = 3.14159265358979323846;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at %s:%d\n", \
                   cudaGetErrorString(err), __FILE__, __LINE__); \
            return -1; \
        } \
    } while(0)


__global__ void zetaKernel(double *partial_sums, const long long n, const double s)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    double sum = 0.0;

    for (long long i = idx + 1; i <= n; i += stride)
    {
        sum += 1.0 / pow(static_cast<double>(i), s);
    }
    
    partial_sums[idx] = sum;
}

__global__ void reduceSumZeta(double *data, int n)
{
    extern __shared__ double sdata[];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? data[i] : 0.0;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0)
    {
        data[blockIdx.x] = sdata[0];
    }
}

double calculateZetaGPU(const double s, const long long n, int threadsPerBlock, int numBlocks)
{
    double *d_partial_sums;
    const int totalThreads = threadsPerBlock * numBlocks;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_partial_sums), totalThreads * sizeof(double)));

    auto *h_result = static_cast<double *>(malloc(totalThreads * sizeof(double)));
    if (!h_result)
    {
        printf("Failed to allocate GPU memory\n");
        return -1;
    }

    zetaKernel<<<numBlocks, threadsPerBlock>>>(d_partial_sums, n, s);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int currentSize = totalThreads;
    while (currentSize > 1)
    {
        int blocks = (currentSize + threadsPerBlock - 1) / threadsPerBlock;
        reduceSumZeta<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_partial_sums, currentSize);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        currentSize = blocks;
    }

    CUDA_CHECK(cudaMemcpy(h_result, d_partial_sums, sizeof(double), cudaMemcpyDeviceToHost));

    const double result = h_result[0];

    cudaFree(d_partial_sums);
    free(h_result);
    
    return result;
}

double calculateZetaCPU(const double s, const long long n)
{
    double sum = 0.0;
    for (long long i = 1; i <= n; i++)
    {
        sum += 1.0 / pow(static_cast<double>(i), s);
    }
    return sum;
}

double getExactZeta(const double s)
{
    if (fabs(s - 2.0) < 1e-10) return M_PI * M_PI / 6.0;
    if (fabs(s - 4.0) < 1e-10) return M_PI * M_PI * M_PI * M_PI / 90.0;
    if (fabs(s - 6.0) < 1e-10) return M_PI * M_PI * M_PI * M_PI * M_PI * M_PI / 945.0;
    return -1.0;
}

int main()
{
    printf("Computation of the Riemann zeta function\n");
    printf("Z(s) = SUM(n=1 to inf) 1/n^s\n\n");

    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0)
    {
        printf("No CUDA devices found!\n");
        return -1;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("The device is used: %s\n\n", prop.name);

    long long n = 10000000;
    int threadsPerBlock = 256;
    int numBlocks = 256;

    printf("Number of row members: %lld\n", n);
    printf("Configuration: %d blocks of %d streams\n\n", numBlocks, threadsPerBlock);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double s_values[] = {2.0, 3.0, 4.0, 5.0, 6.0};
    int num_tests = 5;

    printf("%-10s %-20s %-15s %-20s %-15s\n",
           "s", "Z(s) GPU", "Time (ms)", "Exact value", "Uncertainty");
    printf("-------------------------------------------------------------------------\n");

    for (int i = 0; i < num_tests; i++)
    {
        double s = s_values[i];

        CUDA_CHECK(cudaEventRecord(start, nullptr));
        double zetaGPU = calculateZetaGPU(s, n, threadsPerBlock, numBlocks);
        CUDA_CHECK(cudaEventRecord(stop, nullptr));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float gpuTime;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

        double exact = getExactZeta(s);

        if (exact > 0)
        {
            double error = fabs(zetaGPU - exact);
            printf("%-10.1f %-20.15f %-15.3f %-20.15f %-15.2e\n",
                   s, zetaGPU, gpuTime, exact, error);
        }
        else
        {
            printf("%-10.1f %-20.15f %-15.3f %-20s %-15s\n",
                   s, zetaGPU, gpuTime, "N/A", "N/A");
        }
    }

    printf("\n");

    printf("Compare GPU vs CPU for s = 2:\n");
    long long n_cpu = n / 10;

    CUDA_CHECK(cudaEventRecord(start, nullptr));
    double zetaGPU = calculateZetaGPU(2.0, n, threadsPerBlock, numBlocks);
    CUDA_CHECK(cudaEventRecord(stop, nullptr));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpuTime;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    clock_t cpu_start = clock();
    double zetaCPU = calculateZetaCPU(2.0, n_cpu);
    clock_t cpu_end = clock();
    double cpuTime = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    printf("GPU (n=%lld): Z(2) = %.15f (time: %.3f ms)\n", n, zetaGPU, gpuTime);
    printf("CPU (n=%lld): Z(2) = %.15f (time: %.3f ms)\n", n_cpu, zetaCPU, cpuTime);
    printf("Exact: Z(2) = pi^2/6 = %.15f\n", M_PI * M_PI / 6.0);
    printf("Acceleration: ~%.1fx\n", cpuTime * 10 / gpuTime);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}