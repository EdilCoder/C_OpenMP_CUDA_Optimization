#ifndef __cuda_cuh__
#define __cuda_cuh__

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "internal/common.h"

/**
 * @brief Calculate the standard deviation of the population described by input/N
 *
 * @param input An array of floating point values
 * @param N The length of the array `input`
 * @return The standard deviation (population) of the array `input`
 */
float cuda_standarddeviation(const float *input, size_t N);
/**
 * @brief Calculate the gradient magnitude of the greyscale image `input` and store it within `output`
 *
 * @param input An array of pixel contrasts of an image with width x height pixels
 * @param output A preallocated array to store the resulting gradient magnitude image with (width-2) x (height-2) pixels.
 * @param width The width of image `input`
 * @param height The height of image `input`
 */
void cuda_convolution(const unsigned char *input, unsigned char *output, size_t width, size_t height);
/**
 * @brief Calculate the boundary index of `keys` and store it within `boundaries`
 *
 * Calculate the first occurrence of each integer within inclusive-exclusive range [0, len_b] within `keys`,
 * and store it at the corresponding index in `boundaries`. Where an integer does not occur within the `keys`,
 * it should be assigned the index of the next integer which does occur within `keys`.
 *
 * @param keys A sorted array of integer keys within the inclusive-exclusive range [0, len_b]
 * @param len_k The length of the array `keys`
 * @param boundaries A preallocated array for to store the resulting boundary index
 * @param len_b The length of the array `boundaries`
 */
void cuda_datastructure(const unsigned int *keys, size_t len_k, unsigned int *boundaries, size_t len_b);


/**
 * Error check function for safe CUDA API calling
 * Wrap all calls to CUDA API functions with CUDA_CALL() to catch errors on failure
 * e.g. CUDA_CALL(cudaFree(myPtr));
 * CUDA_CHECk() can also be used to perform error checking after kernel launches and async methods
 * e.g. CUDA_CHECK()
 */
#if defined(_DEBUG) || defined(D_DEBUG)
#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CUDA_CHECK() { gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__); }
#else
#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CUDA_CHECK() { gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__); }
#endif
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        if (line >= 0) {
            fprintf(stderr, "CUDA Error: %s(%d): %s", file, line, cudaGetErrorString(code));
        } else {
            fprintf(stderr, "CUDA Error: %s(%d): %s", file, line, cudaGetErrorString(code));
        }
        exit(EXIT_FAILURE);
    }
}

#endif // __cuda_cuh__
