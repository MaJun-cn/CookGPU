#include <stdio.h>
#include <sys/time.h>

double CpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);

}

int CpuNormalCal(int* data, const int size) {
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += data[i];
    }
    return sum;
}

int CpuRecusiveReduce(int* data, const int size) {
    //terminal check
    if (size == 1)  return data[0];
    const int stride = size / 2;
    for (int i = 0; i < stride; ++i) {
        data[i] += data[i + stride];
    }
    return CpuRecusiveReduce(data, stride);
}

__global__ void GpuReduceNeighbored(int* g_idata, int* g_odata, unsigned int n) {
    // thread id in courrent block
    unsigned int tid = threadIdx.x;
    // id of all thread in grid
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // boundary check
    if (idx >= n) return;
    // subarray in current block
    int *indata = g_idata + blockIdx.x * blockDim.x;
    // cal sum
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            indata[tid] += indata[tid + stride];
        }
        __syncthreads();
    }
    //write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = indata[0];
}

__global__ void GpuReduceNeighboredV2(int* g_idata, int* g_odata, unsigned int n) {
    // thread id in courrent block
    unsigned int tid = threadIdx.x;
    // id of all thread in grid
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // boundary check
    if (idx >= n) return;
    // subarray in current blocks
    int *indata = g_idata + blockIdx.x * blockDim.x;
    // cal sum
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // convert tid into local array index (in block)
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            indata[index] += indata[index + stride];
        }
        __syncthreads();
    }
    //write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = indata[0];
}
__global__ void GpuReduceInterleaved(int* g_idata, int* g_odata, unsigned int n) {
    // thread id in courrent block
    unsigned int tid = threadIdx.x;
    // id of all thread in grid
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // boundary check
    if (idx >= n) return;
    // subarray in current block
    int *indata = g_idata + blockIdx.x * blockDim.x;
    // cal sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            indata[tid] += indata[tid + stride];
        }
        __syncthreads();
    }
    //write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = indata[0];
}
int main(int argc, char** argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("device[%d]: %s\n", dev, deviceProp.name);
    int block_size = 512;
    if (argc > 1) {
        block_size = atoi(argv[1]);
    }
    int size = 1 << 24;
    printf("array size: %d\n", size);
    
    dim3 block(block_size, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("kernal size: grid(%d, %d), block(%d, %d)\n", grid.x, grid.y, block.x, block.y);

    // alloc mem
    size_t bytes = size * sizeof(int);
    int* h_idata = (int*)malloc(bytes);
    int* h_odata = (int*)malloc(grid.x * sizeof(int));
    int* tmp = (int*)malloc(bytes);
    // initialize array
    for (int i = 0; i < size; ++i) {
        h_idata[i] = (int) (rand() & 0xFF);
    }
    // alloc hbm
    int* d_idata = NULL;
    int* d_odata = NULL;
    cudaMalloc((void**) &d_idata, bytes);
    cudaMalloc((void**) &d_odata, grid.x * sizeof(int));

    int gpu_sum = 0;
    // ------ kernal 1 ------
    // copy input data from h to d
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    // cuda kernal cal
    double t1 = CpuSecond();
    GpuReduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    double elaps1 = CpuSecond() - t1;
    // copy output data from d to h
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    // cpu cal
    for (int i = 0; i < grid.x; ++i) {
        gpu_sum += h_odata[i];
    }
    printf("GpuReduceNeighbored result: %d, kernal elaps: %f\n", gpu_sum, elaps1);
    memset(h_odata, 0, grid.x * sizeof(int));
    gpu_sum = 0;

    // ------ kernal 2 ------
    // copy input data from h to d
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    // cuda kernal cal
    double t2 = CpuSecond();
    GpuReduceNeighboredV2<<<grid, block>>>(d_idata, d_odata, size);
    double elaps2 = CpuSecond() - t2;
    // copy output data from d to h
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    // cpu cal
    for (int i = 0; i < grid.x; ++i) {
        gpu_sum += h_odata[i];
    }
    printf("GpuReduceNeighboredV2 result: %d, kernal elaps: %f\n", gpu_sum, elaps2);
    memset(h_odata, 0, grid.x * sizeof(int));
    gpu_sum = 0;

    // ------ kernal 3 ------
    // copy input data from h to d
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    // cuda kernal cal
    double t3 = CpuSecond();
    GpuReduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    double elaps3 = CpuSecond() - t3;
    // copy output data from d to h
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    // cpu cal
    for (int i = 0; i < grid.x; ++i) {
        gpu_sum += h_odata[i];
    }
    double elaps_all_3 = CpuSecond() - t3;
    printf("GpuReduceInterleaved result: %d, kernal elaps: %f, all elaps: %f\n", gpu_sum, elaps3, elaps_all_3);
    memset(h_odata, 0, grid.x * sizeof(int));
    gpu_sum = 0;

    memcpy(tmp, h_idata, bytes);
    // ------ cpu 1 ------
    double t4 = CpuSecond();
    int cpu_sum1 = CpuNormalCal(tmp, size);
    double elaps_all_4 = CpuSecond() - t4;
    // ------ cpu 2 ------
    double t5 = CpuSecond();
    int cpu_sum2 = CpuRecusiveReduce(tmp, size);
    double elaps_all_5 = CpuSecond() - t5;
    printf("cpu normal result: %d, elaps_all: %f\n", cpu_sum1, elaps_all_4);
    printf("cpu recusize result: %d， elaps_all: %f\n", cpu_sum2, elaps_all_5);

    // free host mem
    free(h_idata);
    free(h_odata);
    // free gpu hbm
    cudaFree(d_idata);
    cudaFree(d_odata);
    // reset device
    cudaDeviceReset();
}
/*
device[0]: Tesla V100-SXM2-32GB
array size: 16777216
kernal size: grid(32768, 1), block(512, 1)
GpuReduceNeighbored result: 2139353471, kernal elaps: 0.000035
GpuReduceNeighboredV2 result: 2139353471, kernal elaps: 0.000017
GpuReduceInterleaved result: 2139353471, kernal elaps: 0.000011, all elaps: 0.000567
cpu normal result: 2139353471, elaps_all: 0.043164
cpu recusize result: 2139353471， elaps_all: 0.042999
*/