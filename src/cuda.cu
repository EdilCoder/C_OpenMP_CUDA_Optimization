#include "cuda.cuh"

#include <cstring>

#define THREADS_PER_BLOCK 256
__global__ void my_kernel(const float* input, size_t N, float* sum, float* sumSq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ float s_Sum[THREADS_PER_BLOCK];
    __shared__ float s_SumSq[THREADS_PER_BLOCK];

    float tempSum = 0.0f;
    float tempSumSq = 0.0f;

    while (i < N) {
        tempSum += input[i];
        tempSumSq += input[i] * input[i];
        i += stride;
    }

    s_Sum[threadIdx.x] = tempSum;
    s_SumSq[threadIdx.x] = tempSumSq;

    __syncthreads();

    int b = blockDim.x / 2;
    while (b != 0) {
        if (threadIdx.x < b) {
            s_Sum[threadIdx.x] += s_Sum[threadIdx.x + b];
            s_SumSq[threadIdx.x] += s_SumSq[threadIdx.x + b];
        }
        __syncthreads();
        b /= 2;
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum, s_Sum[0]);
        atomicAdd(sumSq, s_SumSq[0]);
    }
}
float cuda_standarddeviation(const float* const input, const size_t N) {
    float* d_input, * d_sum, * d_sumSq;
    float sum = 0.0f, sumSq = 0.0f;

    CUDA_CALL(cudaMalloc((void**)&d_input, N * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_sum, sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_sumSq, sizeof(float)));

    CUDA_CALL(cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemset(d_sum, 0, sizeof(float)));
    CUDA_CALL(cudaMemset(d_sumSq, 0, sizeof(float)));

    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    my_kernel << <numBlocks, THREADS_PER_BLOCK >> > (d_input, N, d_sum, d_sumSq);

    CUDA_CALL(cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&sumSq, d_sumSq, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_input));
    CUDA_CALL(cudaFree(d_sum));
    CUDA_CALL(cudaFree(d_sumSq));
    //cudaDeviceReset();

    float mean = sum / N;
    float variance = sumSq / N - mean * mean;

    return sqrtf(variance);
}

__global__ void sobel_kernel(const unsigned char* const input, unsigned char* const output, const size_t width, const size_t height) {

    __shared__ int horizontal_sobel[3][3];
    __shared__ int vertical_sobel[3][3];

    if (threadIdx.x == 0 && threadIdx.y == 0) {

        horizontal_sobel[0][0] = 1; horizontal_sobel[0][1] = 0; horizontal_sobel[0][2] = -1;
        horizontal_sobel[1][0] = 2; horizontal_sobel[1][1] = 0; horizontal_sobel[1][2] = -2;
        horizontal_sobel[2][0] = 1; horizontal_sobel[2][1] = 0; horizontal_sobel[2][2] = -1;

        vertical_sobel[0][0] = 1; vertical_sobel[0][1] = 2; vertical_sobel[0][2] = 1;
        vertical_sobel[1][0] = 0; vertical_sobel[1][1] = 0; vertical_sobel[1][2] = 0;
        vertical_sobel[2][0] = -1; vertical_sobel[2][1] = -2; vertical_sobel[2][2] = -1;
    }
    __syncthreads();

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >  0 && x < width - 1 && y > 0 && y < height - 1) {
        unsigned int g_x = 0;
        unsigned int g_y = 0;

        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                const size_t input_offset = (y + i) * width + (x + j);
                g_x += input[input_offset] * horizontal_sobel[j + 1][i + 1];
                g_y += input[input_offset] * vertical_sobel[j + 1][i + 1];
            }
        }

        const size_t output_offset = (y - 1) * (width - 2) + (x - 1);
        const float grad_mag = sqrtf((float)((g_x * g_x) + (g_y * g_y)));
        output[output_offset] = (size_t)(grad_mag / 3 > 255 ? 255 : grad_mag / 3);
    }
}
void cuda_convolution(const unsigned char* const input, unsigned char* const output, const size_t width, const size_t height) {

    unsigned char* d_input, * d_output;
    cudaMalloc((void**)&d_input, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, (width - 2) * (height - 2) * sizeof(unsigned char));
    cudaMemcpy(d_input, input, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width - 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height - 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sobel_kernel<<<numBlocks, threadsPerBlock>>> (d_input, d_output, width, height);

    cudaMemcpy(output, d_output, (width - 2) * (height - 2) * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}


__global__ void histogram_kernel(const unsigned int* keys, size_t len_k, unsigned int* histogram) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len_k) {
        atomicAdd(&histogram[keys[idx]], 1);
    }
}
void cuda_datastructure(const unsigned int* const keys, const size_t len_k, unsigned int* const boundaries, const size_t len_b) {
    unsigned int* d_keys, * d_boundaries, * d_histogram;
    size_t histogram_bytes = (len_b - 1) * sizeof(unsigned int);

    CUDA_CALL(cudaMalloc((void**)&d_keys, len_k * sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc((void**)&d_boundaries, len_b * sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc((void**)&d_histogram, histogram_bytes));

    CUDA_CALL(cudaMemset(d_histogram, 0, histogram_bytes));
    CUDA_CALL(cudaMemset(d_boundaries, 0, len_b * sizeof(unsigned int)));
    CUDA_CALL(cudaMemcpy(d_keys, keys, len_k * sizeof(unsigned int), cudaMemcpyHostToDevice));

    int blockSize = 512; 
    int numBlocks = (len_k + blockSize - 1) / blockSize;
    histogram_kernel << <numBlocks, blockSize >> > (d_keys, len_k, d_histogram);

    CUDA_CHECK();  

    unsigned int* histogram = (unsigned int*)malloc(histogram_bytes);
    CUDA_CALL(cudaMemcpy(histogram, d_histogram, histogram_bytes, cudaMemcpyDeviceToHost));

    boundaries[0] = 0;
    for (size_t i = 0; i < len_b - 1; ++i) {
        boundaries[i + 1] = boundaries[i] + histogram[i];
    }

    free(histogram);
    CUDA_CALL(cudaFree(d_keys));
    CUDA_CALL(cudaFree(d_boundaries));
    CUDA_CALL(cudaFree(d_histogram));
}







