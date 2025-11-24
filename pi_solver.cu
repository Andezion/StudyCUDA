#include <cstdio>
#include <cuda_runtime.h>

double M_PI = 3.14159265358979323846;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "Error CUDA in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void computeIntegral(double *partial_sums, const long long n, const double dx)
{
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = blockDim.x * gridDim.x;

    double local_sum = 0.0;

    for (long long i = idx; i < n; i += stride)
    {
        const double x = (i + 0.5) * dx;
        local_sum += sqrt(1.0 - x * x);
    }

    partial_sums[idx] = local_sum * dx;
}

double reduceSums(const double *partial_sums, const int size)
{
    double total = 0.0;
    for (int i = 0; i < size; i++)
    {
        total += partial_sums[i];
    }
    return total;
}

int main() {
    constexpr long long n = 100000000;
    constexpr double dx = 1.0 / n;

    int threadsPerBlock = 256;
    int blocksPerGrid = 1024;
    const int totalThreads = threadsPerBlock * blocksPerGrid;

    printf("Calculating the number pi by integration\n");
    printf("Number of splits: %lld\n", n);
    printf("Blocks: %d, Threads per block: %d\n\n", blocksPerGrid, threadsPerBlock);

    auto *h_partial_sums = static_cast<double *>(malloc(totalThreads * sizeof(double)));
    if (h_partial_sums == nullptr)
    {
        fprintf(stderr, "Memory allocation error on host\n");
        return EXIT_FAILURE;
    }

    double *d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, totalThreads * sizeof(double)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    computeIntegral<<<blocksPerGrid, threadsPerBlock>>>(d_partial_sums, n, dx);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums,
                          totalThreads * sizeof(double),
                          cudaMemcpyDeviceToHost));

    const double integral = reduceSums(h_partial_sums, totalThreads);

    const double pi_calculated = 4.0 * integral;
    const double pi_actual = M_PI;
    const double error = fabs(pi_calculated - pi_actual);

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Results:\n");
    printf("  Calculated value pi: %.15f\n", pi_calculated);
    printf("  Exact value pi:       %.15f\n", pi_actual);
    printf("  Absolute error:  %.15e\n", error);
    printf("  Relative error: %.10e%%\n", (error / pi_actual) * 100);
    printf("\nLead time on GPU: %.3f ms\n", milliseconds);

    free(h_partial_sums);
    CUDA_CHECK(cudaFree(d_partial_sums));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\nProgram completed successfully!\n");

    return EXIT_SUCCESS;
}