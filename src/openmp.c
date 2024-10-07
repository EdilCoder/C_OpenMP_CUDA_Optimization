#include "openmp.h"

#include <stdlib.h>
#include <string.h>
#include <omp.h>

float openmp_standarddeviation(const float* const input, const size_t N) {
    float sum = 0.0f;
    float sumSq = 0.0f;
    size_t i;

#pragma omp parallel for reduction(+:sum, sumSq) schedule(guided,64) num_threads(16)
    for (i = 0; i < N; ++i) {
        sum += input[i];
        sumSq += input[i] * input[i];
    }

    float mean = sum / N;
    float variance = sumSq / N - mean * mean;

    return sqrtf(variance);
}

inline char clamp(const float v, const float lo, const float hi) {
    return (char)(v < lo ? lo : v > hi ? hi : v);
}

void openmp_convolution(const unsigned char* const input, unsigned char* const output, const size_t width, const size_t height) {
    const int horizontal_sobel[3][3] = {
        { 1, 0,-1},
        { 2, 0,-2},
        { 1, 0,-1}
    };
    const int vertical_sobel[3][3] = {
        { 1, 2, 1},
        { 0, 0, 0},
        {-1,-2,-1}
    };
    size_t x;
    size_t y;
#pragma omp parallel for collapse(2) private(x,y) schedule(guided,4) num_threads(64)
    for (x = 1; x < width - 1; ++x) {
        for (y = 1; y < height - 1; ++y) {
            unsigned int g_x = 0;
            unsigned int g_y = 0;
            //---optimize_2---
            size_t input_offset = (y - 1) * width + (x - 1);
            const unsigned char* input_ptr = &input[input_offset];

            g_x += input_ptr[0] * horizontal_sobel[0][0] + input_ptr[1] * horizontal_sobel[0][1] + input_ptr[2] * horizontal_sobel[0][2];
            g_y += input_ptr[0] * vertical_sobel[0][0] + input_ptr[1] * vertical_sobel[0][1] + input_ptr[2] * vertical_sobel[0][2];
            input_ptr += width;
            g_x += input_ptr[0] * horizontal_sobel[1][0] + input_ptr[1] * horizontal_sobel[1][1] + input_ptr[2] * horizontal_sobel[1][2];
            g_y += input_ptr[0] * vertical_sobel[1][0] + input_ptr[1] * vertical_sobel[1][1] + input_ptr[2] * vertical_sobel[1][2];
            input_ptr += width;
            g_x += input_ptr[0] * horizontal_sobel[2][0] + input_ptr[1] * horizontal_sobel[2][1] + input_ptr[2] * horizontal_sobel[2][2];
            g_y += input_ptr[0] * vertical_sobel[2][0] + input_ptr[1] * vertical_sobel[2][1] + input_ptr[2] * vertical_sobel[2][2];

            //optimize_1
            /*for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    const size_t input_offset = (y + i) * width + (x + j);
                    g_x += input[input_offset] * horizontal_sobel[j + 1][i + 1];
                    g_y += input[input_offset] * vertical_sobel[j + 1][i + 1];
                }
            }*/
            const size_t output_offset = (y - 1) * (width - 2) + (x - 1);
            const float grad_mag = sqrtf((float)((g_x * g_x) + (g_y * g_y)));
            output[output_offset] = clamp(grad_mag / 3, 0.0f, 255.0f);
        }
    }
}

void openmp_datastructure(const unsigned int* const keys, const size_t len_k, unsigned int* const boundaries, const size_t len_b) {

    const size_t histogram_bytes = (len_b - 1) * sizeof(unsigned int);
    unsigned int* histogram = (unsigned int*)malloc(histogram_bytes);
    memset(histogram, 0, histogram_bytes);
    size_t i;

#pragma omp parallel for shared(histogram) private(i) num_threads(1024)
    for (i = 0; i < len_k; ++i) {
        ++histogram[keys[i]];
    }

    memset(boundaries, 0, len_b * sizeof(unsigned int));
    for (i = 0; i < len_b - 1; ++i) {
        boundaries[i + 1] = boundaries[i] + histogram[i];
    }

    free(histogram);
}
