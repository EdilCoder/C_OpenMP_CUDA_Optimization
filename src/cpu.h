#ifndef __cpu_h__
#define __cpu_h__

#include <stddef.h>

#include "internal/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate the standard deviation of the population described by input/N
 *
 * @param input An array of floating point values
 * @param N The length of the array `input`
 * @return The standard deviation (population) of the array `input`
 */
float cpu_standarddeviation(const float *input, size_t N);
/**
 * @brief Calculate the gradient magnitude of the greyscale image `input` and store it within `output`
 *
 * @param input An array of pixel contrasts of an image with width x height pixels
 * @param output A preallocated array to store the resulting gradient magnitude image with (width-2) x (height-2) pixels.
 * @param width The width of image `input`
 * @param height The height of image `input`
 */
void cpu_convolution(const unsigned char *input, unsigned char *output, size_t width, size_t height);
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
void cpu_datastructure(const unsigned int *keys, size_t len_k, unsigned int *boundaries, size_t len_b);

#ifdef __cplusplus
}
#endif

#endif // __cpu_h__
