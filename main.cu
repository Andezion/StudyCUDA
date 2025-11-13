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
    hello_world<<<1, 1>>> ();
    hello_world_2<<<1, 1>>> ();
    getchar();
    return 0;
}
