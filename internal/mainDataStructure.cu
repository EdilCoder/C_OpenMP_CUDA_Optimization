#include <cstdlib>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <random>

#include "config.h"
#include "common.h"
#include "cpu.h"
#include "openmp.h"
#include "cuda.cuh"
#include "main.h"

namespace {
/**
 * Structure containing the options provided by runtime arguments
 */
struct DSConfig {
    /**
      * Path to input file
      */
    char *input_file = nullptr;
    /**
      * Path to output file
      */
    char *output_file = nullptr;
    /**
      * Random seed
      * If neither input_file nor random_seed are used, time will be used as seed
      */
    unsigned int random_seed = 0;
    /**
      * Random length
      * If generating random data, this is how much data will be generated
      */
    unsigned int random_length = 1000000;
    /**
      * Program will operate in benchmark mode
      * This repeats the algorithm multiple times and returns an average time
      */
    bool benchmark;
}; typedef struct DSConfig DSConfig;
/**
 * Parse the runtime args into config
 * @param argc argc from main()
 * @param argv argv from main()]
 * @param config Pointer to config structure for return value
 */
void parse_args(int argc, char** argv, DSConfig* config) {
    // Clear config struct
    *config = {};
    // Iterate over remaining args    
    int i = 3;
    char* t_arg = 0;
    for (; i < argc; i++) {
        // Make a lowercase copy of the argument
        const size_t arg_len = strlen(argv[i]) + 1;  // Add 1 for null terminating character
        if (t_arg)
            free(t_arg);
        t_arg = (char*)malloc(arg_len);
        int j = 0;
        for (; argv[i][j]; ++j) {
            t_arg[j] = tolower(argv[i][j]);
        }
        t_arg[j] = '\0';
        // Decide which arg it is
        // Benchmark
        if (!strcmp("--bench", t_arg) || !strcmp("--benchmark", t_arg) || !strcmp("-b", t_arg)) {
            config->benchmark = 1;
            continue;
        }
        // Input file
        if (!strcmp(t_arg + arg_len - 5, ".csv")) {
            if (config->input_file  || (config->random_seed && config->random_length)) {
                if (config->output_file) {
                    fprintf(stderr, "Multiple inputs/outputs were provided, this is not supported!\n");
                    print_help(argv[0]);
                } else {
                    // Allocate memory and copy
                    config->output_file = (char*)malloc(arg_len);
                    memcpy(config->output_file, argv[i], arg_len);
                    continue;
                }
            } else {
                // Allocate memory and copy
                config->input_file = (char*)malloc(arg_len);
                memcpy(config->input_file, argv[i], arg_len);
                continue;
            }
        }
        // Random seed + length
        if (i + 1 < argc) {
            // Random seed
            char* end = nullptr;
            const unsigned int t_arg_uint = (unsigned int)strtoul(argv[i], &end, 10);
            // Test that it converts back to the same string as a form of validation
            const int n = snprintf(NULL, 0, "%u", t_arg_uint);
            if (n > 0) {
                char* buf = (char*)malloc(n + 1);
                int c = snprintf(buf, n + 1, "%u", t_arg_uint);
                if (!strcmp(buf, argv[i])) {
                    if (config->random_seed) {
                        fprintf(stderr, "Multiple random seeds were provided, this is not supported!\n");
                        print_help(argv[0]);
                    }
                    config->random_seed = t_arg_uint;
                }
                free(buf);
            }
            if (config->random_seed) {  // Length
                ++i;
                char* end = nullptr;
                const unsigned int t_arg_uint = (unsigned int)strtoul(argv[i], &end, 10);
                // Test that it converts back to the same string as a form of validation
                const int n = snprintf(NULL, 0, "%u", t_arg_uint);
                if (n > 0) {
                    char* buf = (char*)malloc(n + 1);
                    int c = snprintf(buf, n + 1, "%u", t_arg_uint);
                    if (!strcmp(buf, argv[i])) {
                        free(buf);
                        config->random_length = t_arg_uint;
                    } else {
                        free(buf);
                    }
                }
            }
        } else {
            fprintf(stderr, "Unexpected standard deviation argument: %s\n", argv[i]);
            print_help(argv[0]);
        }
    }
    if (config->input_file && config->random_seed) {
        fprintf(stderr, "Both input file and random seed were specified\n");
        print_help(argv[0]);
    } else if (!config->input_file && !config->random_seed) {
        fprintf(stderr, "Neither input file nor random seed/length were specified\n");
        print_help(argv[0]);
    }
    if (t_arg)
        free(t_arg);
}
}

void runDataStructure(int argc, char** argv, const Implementation implementation) {
    DSConfig config;
    parse_args(argc, argv, &config);

    // Inputs
    unsigned int *input_keys = nullptr;
    size_t input_keys_elements = 0;

    // Load/Generate input
    if (config.input_file) {
        // Load CSV
        printf("Using input file: %s%s%s\n", CONSOLE_YELLOW, config.input_file, CONSOLE_RESET);
        loadCSV(config.input_file, reinterpret_cast<void**>(&input_keys), &input_keys_elements, "%u");
        printf("Input has length: %s%u%s\n", CONSOLE_YELLOW, static_cast<unsigned int>(input_keys_elements), CONSOLE_RESET);
    } else {
        // Random init
        if (!config.random_seed) {
            config.random_seed = static_cast<unsigned int>(time(nullptr));
        }
        printf("Using random seed: %s%u%s\n", CONSOLE_YELLOW, config.random_seed, CONSOLE_RESET);
        printf("Generating input of length: %s%u%s\n", CONSOLE_YELLOW, config.random_length, CONSOLE_RESET);
        // Generate a random population
        input_keys_elements = config.random_length;
        input_keys = static_cast<unsigned int*>(malloc(input_keys_elements * sizeof(unsigned int)));
        std::mt19937 rng(config.random_seed);
        std::normal_distribution<float> dist(0.0, 10.0);
        unsigned int num = 0;
        for (unsigned int i = 0; i < input_keys_elements;) {
            const unsigned int count = static_cast<unsigned int>(abs(floor(dist(rng))));
            for (unsigned int j = 0; j < count && i < input_keys_elements; ++j) {
                input_keys[i++] = num;
            }
            ++num;            
        }
    }

    // Create result for validation
    const size_t boundaries_elements = input_keys[input_keys_elements - 1] + 2;
    unsigned int* validation_boundaries = static_cast<unsigned int*>(malloc(boundaries_elements * sizeof(unsigned int)));
    cpu_datastructure(input_keys, input_keys_elements, validation_boundaries, boundaries_elements);

    // Run student implementation
    float timing_log;
    unsigned int* result_boundaries = static_cast<unsigned int*>(malloc(boundaries_elements * sizeof(unsigned int)));
    const int TOTAL_RUNS = config.benchmark ? BENCHMARK_RUNS : 1;
    {
        //Init for run  
        cudaEvent_t startT, stopT;
        CUDA_CALL(cudaEventCreate(&startT));
        CUDA_CALL(cudaEventCreate(&stopT));
        // Run 1 or many times
        timing_log = 0.0f;
        for (int runs = 0; runs < TOTAL_RUNS; ++runs) {
            if (TOTAL_RUNS > 1)
                printf("\r%d/%d", runs + 1, TOTAL_RUNS);
            // Run Adaptive Histogram algorithm
            CUDA_CALL(cudaEventRecord(startT));
            CUDA_CALL(cudaEventSynchronize(startT));
            switch (implementation) {
            case CPU:
                cpu_datastructure(input_keys, input_keys_elements, result_boundaries, boundaries_elements);
                break;
            case OPENMP:
                openmp_datastructure(input_keys, input_keys_elements, result_boundaries, boundaries_elements);
                break;
            case CUDA:
                cuda_datastructure(input_keys, input_keys_elements, result_boundaries, boundaries_elements);
                break;
            }
            CUDA_CALL(cudaEventRecord(stopT));
            CUDA_CALL(cudaEventSynchronize(stopT));
            // Sum timing info
            float milliseconds = 0;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, startT, stopT));
            timing_log += milliseconds;
        }
        if (TOTAL_RUNS > 1)
            printf("\n");
        // Convert timing info to average
        timing_log /= TOTAL_RUNS;

        // Cleanup timing
        cudaEventDestroy(startT);
        cudaEventDestroy(stopT);
    }

    // Validate and report
    {
        unsigned int errors = 0;
        for (unsigned int i = 0; i < boundaries_elements; ++i) {
            if (validation_boundaries[i] != result_boundaries[i]) {
                ++errors;
            }
        }
        printf("Date Structure Result: %s" CONSOLE_RESET "\n", errors ? CONSOLE_RED "Fail" : CONSOLE_GREEN "Pass");
        if (errors) {
            printf("\t%u/%u elements wrong!" CONSOLE_RESET "\n", errors, static_cast<unsigned int>(boundaries_elements));
            printf("\t(Consider comparing output csvs)\n");
        }
    }

    // Export output
    if (config.output_file) {
        saveCSV(config.output_file, result_boundaries, boundaries_elements);
    }
    
    // Report timing information    
    printf("%s average execution timing from %d runs\n", implementation_to_string(implementation), TOTAL_RUNS);
    if (implementation == CUDA) {
        int device_id = 0;
        CUDA_CALL(cudaGetDevice(&device_id));
        cudaDeviceProp props;
        memset(&props, 0, sizeof(cudaDeviceProp));
        CUDA_CALL(cudaGetDeviceProperties(&props, device_id));
        printf("Using GPU: %s\n", props.name);
    }
#ifdef _DEBUG
    printf(CONSOLE_YELLOW "Code built as DEBUG, timing results are invalid!\n" CONSOLE_RESET);
#endif
    printf("Time: %.3fms\n", timing_log);

    // Cleanup
    free(input_keys);
    free(validation_boundaries);
    free(result_boundaries);
    if (config.output_file)
        free(config.output_file);
    if (config.input_file)
        free(config.input_file);
}
