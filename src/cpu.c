#include "cpu.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

inline char clamp(const float v, const float lo, const float hi) {
    return (char)(v < lo ? lo : v > hi ? hi : v);
}

float cpu_standarddeviation(const float *const input, const size_t N) {
    // Calculate the mean of input
    float mean = 0;
    for (size_t i = 0; i < N; ++i) {
        mean += input[i];
    }
    mean /= (float)N;
    // Subtract the mean from input and square the result
    float* devmeansq = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; ++i) {
        devmeansq[i] = powf(input[i] - mean, 2.0f);
    }
    // Calculate the sum of devmeansq
    float sum = 0;
    for (size_t i = 0; i < N; ++i) {
        sum += devmeansq[i];
    }
    // Cleanup
    free(devmeansq);
    // Return the standard deviation
    return sqrtf(sum / (float)N);
}
void cpu_convolution(const unsigned char *const input, unsigned char *const output, const size_t width, const size_t height) {
    const int horizontal_sobel[3][3] = {
        { 1, 0,-1},
        { 2, 0,-2},
        { 1, 0,-1}};
    const int vertical_sobel[3][3] = {
        { 1, 2, 1},
        { 0, 0, 0},
        {-1,-2,-1}};
    // For each non-boundary input pixel
    for (size_t x = 1; x + 1 < width; ++x) {
        for (size_t y = 1; y + 1 < height; ++y) {
            unsigned int g_x = 0;
            unsigned int g_y = 0;
            // For each pixel in moore neighbourhood
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    // Calculate offset into input
                    const size_t input_offset = (y + i) * width + (x + j);
                    g_x += input[input_offset] * horizontal_sobel[j + 1][i + 1];
                    g_y += input[input_offset] * vertical_sobel[j + 1][i + 1];
                }
            }
            // Calculate offset into output
            const size_t output_offset = (y-1) * (width - 2) + (x - 1);
            // Calculate gradient magnitude
            const float grad_mag = sqrtf((float)((g_x * g_x) + (g_y * g_y)));
            // Technically we should normalize so pixels are scaled to [0, 255]
            // Instead just divide by 3 and clamp as an approximate method
            output[output_offset] = clamp(grad_mag / 3, 0.0f, 255.0f);
        }
    }
}
void cpu_datastructure(const unsigned int *const keys, const size_t len_k, unsigned int *const boundaries, const size_t len_b) {
    // Allocate Histogram
    const size_t histogram_bytes = (len_b - 1) * sizeof(unsigned int);
    unsigned int *histogram = (unsigned int*)malloc(histogram_bytes);
    memset(histogram, 0, histogram_bytes);
    // Calculate Histogram
    for (size_t i = 0; i < len_k; ++i) {
        ++histogram[keys[i]];
    }
    // Calculate Boundary Index
    memset(boundaries, 0, len_b * sizeof(unsigned int));
    for (size_t i = 0; i + 1 < len_b; ++i) {
        boundaries[i + 1] = boundaries[i] + histogram[i];
    }
    // Free Histogram
    free(histogram);
}
