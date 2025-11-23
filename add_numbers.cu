#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

__global__ void add( const int *a, const int *b, int *c )
{
    *c = *a + *b;
}

__global__ void sub( const int *a, const int *b, int *c)
{
    *c = *a - *b;
}

__global__ void mul( const int *a, const int *b, int *c)
{
    *c = *a * *b;
}



int main()
{

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, nullptr);


    int a, b, c;

    int *dev_a, *dev_b, *dev_c;
    constexpr int size = sizeof( int );

    cudaMalloc( reinterpret_cast<void **>(&dev_a), size );
    cudaMalloc( reinterpret_cast<void **>(&dev_b), size );
    cudaMalloc( reinterpret_cast<void **>(&dev_c), size );

    a = 2;
    b = 7;

    cudaMemcpy( dev_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_b, &b, size, cudaMemcpyHostToDevice);

    add<<< 1, 1 >>>( dev_a, dev_b, dev_c );
    cudaMemcpy( &c, dev_c, size, cudaMemcpyDeviceToHost);

    printf("%d + %d = %d\n", a, b, c);
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_c );


    cudaEventRecord(stop,nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\nTime spent executing by the GPU: %.2fmillseconds\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}