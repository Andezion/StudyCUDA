#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>

double M_PI = 3.14159265358979323846;

#define CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)


__device__ unsigned int rng_step(unsigned int &state)
{
    unsigned int x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

__device__ float rng_uniform01(unsigned int &state)
{
    const unsigned int r = rng_step(state);
    return (r + 0.5f) * (1.0f / 4294967296.0f);
}

__device__ unsigned int rng_init(const unsigned long long seed, const unsigned int tid)
{
    unsigned long long s = seed;
    s ^= static_cast<unsigned long long>(tid) * 0x9e3779b97f4a7c15ULL;
    s = (s ^ s >> 33) * 0xff51afd7ed558ccdULL;
    auto st = static_cast<unsigned int>(s >> 16 & 0xffffffffu);

    if (st == 0)
    {
        st = 0x9e3779b9u;
    }
    return st;
}

__global__ void kernel_naive(unsigned long long *global_hits,
                             const unsigned long long samples_per_thread,
                             const unsigned long long seed)
{
    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int bdx = blockDim.x;
    const unsigned int bdy = blockDim.y;
    const unsigned int gridx = gridDim.x;

    const unsigned int i = bx * bdx + tx;
    const unsigned int j = by * bdy + ty;
    const unsigned int width = gridx * bdx;
    const unsigned int tid = i + j * width;

    unsigned int state = rng_init(seed, tid);

    for (unsigned long long s = 0; s < samples_per_thread; ++s)
    {
        const float x = rng_uniform01(state);
        const float y = rng_uniform01(state);

        if (x * x + y * y <= 1.0f)
        {
            atomicAdd(global_hits, 1ull);
        }
    }
}

__global__ void kernel_block_reduce(unsigned long long *global_hits,
                                    const unsigned long long samples_per_thread,
                                    const unsigned long long seed)
{
    extern __shared__ unsigned int sdata[];
    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int bdx = blockDim.x;
    const unsigned int bdy = blockDim.y;
    const unsigned int gridx = gridDim.x;

    const unsigned int i = bx * bdx + tx;
    const unsigned int j = by * bdy + ty;
    const unsigned int width = gridx * bdx;
    const unsigned int tid = i + j * width;

    unsigned int state = rng_init(seed, tid);

    unsigned int local_hits = 0u;
    for (unsigned long long s = 0; s < samples_per_thread; ++s)
    {
        const float x = rng_uniform01(state);
        const float y = rng_uniform01(state);

        if (x * x + y * y <= 1.0f)
        {
            local_hits++;
        }
    }

    const unsigned int local_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int block_threads = blockDim.x * blockDim.y;

    sdata[local_tid] = local_hits;
    __syncthreads();

    for (unsigned int stride = (block_threads + 1) / 2; stride > 0; stride = (stride + 1) / 2)
    {
        if (local_tid < stride && local_tid + stride < block_threads)
        {
            sdata[local_tid] += sdata[local_tid + stride];
        }
        __syncthreads();
        if (stride == 1) break;
    }

    if (local_tid == 0)
    {
        atomicAdd(global_hits, sdata[0]);
    }
}


int main(const int argc, char **argv)
{
    if (argc < 6)
    {
        printf("Usage: %s grid_x grid_y block_x block_y samples_per_thread [mode]\n", argv[0]);
        printf(" mode: 0 = naive (atomic per sample), 1 = block-reduce (one atomic per block)\n");
        return 0;
    }

    const int grid_x = atoi(argv[1]);
    const int grid_y = atoi(argv[2]);
    const int block_x = atoi(argv[3]);
    const int block_y = atoi(argv[4]);

    const unsigned long long samples_per_thread = strtoull(argv[5], nullptr, 10);
    int mode = 1;

    if (argc >= 7)
    {
        mode = atoi(argv[6]);
    }

    dim3 grid(grid_x, grid_y);
    dim3 block(block_x, block_y);

    const size_t threads_total = static_cast<size_t>(grid_x) * grid_y * block_x * block_y;
    const unsigned long long total_samples = threads_total * samples_per_thread;

    printf("Grid: %dx%d, Block: %dx%d, threads total: %zu\n", grid_x, grid_y, block_x, block_y, threads_total);
    printf("Samples per thread: %llu, total samples: %llu\n", samples_per_thread, total_samples);
    printf("Mode: %s\n", (mode == 0) ? "naive" : "block-reduce");

    unsigned long long *d_hits;

    CHECK(cudaMalloc(&d_hits, sizeof(unsigned long long)));
    CHECK(cudaMemset(d_hits, 0, sizeof(unsigned long long)));

    const unsigned long long seed = 1469598103934665603ULL ^ static_cast<unsigned long long>(time(nullptr));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(start, nullptr));
    if (mode == 0)
    {
        kernel_naive<<<grid, block>>>(d_hits, samples_per_thread, seed);
    }
    else
    {
        const size_t block_threads = block_x * block_y;
        size_t shared_bytes = block_threads * sizeof(unsigned int);
        kernel_block_reduce<<<grid, block, shared_bytes>>>(d_hits, samples_per_thread, seed);
    }
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop, nullptr));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    unsigned long long h_hits = 0;
    CHECK(cudaMemcpy(&h_hits, d_hits, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    const double fraction = static_cast<double>(h_hits) / static_cast<double>(total_samples);
    double pi_est = fraction * 4.0;
    const double error = fabs(M_PI - pi_est);

    printf("Hits: %llu\n", h_hits);
    printf("Estimated pi = %.10f, error = %.10e\n", pi_est, error);
    printf("Elapsed kernel time: %.3f ms\n", ms);

    CHECK(cudaFree(d_hits));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return 0;
}
