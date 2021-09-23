#include <stdio.h>
#include <sys/time.h>

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
    GpuReduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    // copy output data from d to h
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    // cpu cal
    for (int i = 0; i < grid.x; ++i) {
        gpu_sum += h_odata[i];
    }
    printf("GpuReduceNeighbored result: %d\n", gpu_sum);
    memset(h_odata, 0, grid.x * sizeof(int));
    gpu_sum = 0;

    // ------ kernal 2 ------
    // copy input data from h to d
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    // cuda kernal cal
    GpuReduceNeighboredV2<<<grid, block>>>(d_idata, d_odata, size);
    // copy output data from d to h
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    // cpu cal
    for (int i = 0; i < grid.x; ++i) {
        gpu_sum += h_odata[i];
    }
    printf("GpuReduceNeighboredV2 result: %d\n", gpu_sum);
    memset(h_odata, 0, grid.x * sizeof(int));
    gpu_sum = 0;

    // ------ kernal 3 ------
    // copy input data from h to d
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    // cuda kernal cal
    GpuReduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    // copy output data from d to h
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    // cpu cal
    for (int i = 0; i < grid.x; ++i) {
        gpu_sum += h_odata[i];
    }
    printf("GpuReduceInterleaved result: %d\n", gpu_sum);
    memset(h_odata, 0, grid.x * sizeof(int));
    gpu_sum = 0;

    memcpy(tmp, h_idata, bytes);
    printf("cpu normal result: %d\n", CpuNormalCal(tmp, size));
    printf("cpu recusize result: %d\n", CpuRecusiveReduce(tmp, size));

    // free host mem
    free(h_idata);
    free(h_odata);
    // free gpu hbm
    cudaFree(d_idata);
    cudaFree(d_odata);
    // reset device
    cudaDeviceReset();
}