#include <iostream>
#include <cuda_runtime.h>

__global__ void add(const int a, const int b, int *c)
{
    printf("Thread %d in block %d: %d + %d = %d\n", 
           threadIdx.x, blockIdx.x, a, b, a + b);
    *c = a + b;
}

int main()
{
    int a, b, c;
    int *dev_c;
    
    std::cout << "Enter first number: ";
    std::cin >> a;
    std::cout << "Enter second number: ";
    std::cin >> b;
    
    cudaMalloc(reinterpret_cast<void **>(&dev_c), sizeof(int));

    add<<<1, 1>>>(a, b, dev_c);

    cudaDeviceSynchronize();
    
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "\nResult: " << a << " + " << b << " = " << c << std::endl;
    
    cudaFree(dev_c);
    
    return 0;
}