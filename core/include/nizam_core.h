#ifndef NIZAM_CORE_H
#define NIZAM_CORE_H

#include <stdint.h>

#ifdef _WIN32
#define NIZAM_API __declspec(dllexport)
#else
#define NIZAM_API
#endif

extern "C" {

    // 8-bit integer quantization helper (scales float values into [-128, 127])
    NIZAM_API void quantize_int8(const float* input, int8_t* output, int size, float scale, int8_t zero_point);
    
    // Dequantize 8-bit back to float
    NIZAM_API void dequantize_int8(const int8_t* input, float* output, int size, float scale, int8_t zero_point);

    // 1-bit binary quantization helper (converts floats to binary weights, positive -> 1, negative -> -1)
    NIZAM_API void quantize_binary(const float* input, int8_t* output, int size);

    // 4-bit integer quantization helper (scales float values into [-8, 7])
    NIZAM_API void quantize_int4(const float* input, int8_t* output, int size, float scale, int8_t zero_point);

    // Dequantize 4-bit back to float
    NIZAM_API void dequantize_int4(const int8_t* input, float* output, int size, float scale, int8_t zero_point);

    // Simulated hardware vector instruction acceleration (e.g. ARM NEON or RISC-V Vector)
    // multiplier: factor simulating acceleration (e.g. 4 for NEON, 8 for RVV)
    NIZAM_API void simulate_hardware_vector_ops(const float* a, const float* b, float* result, int size, int hardware_type);

    // Simple fast logic clause evaluation. Evaluates a conjunction of features against thresholds.
    // returns 1 if all conditions satisfied, 0 otherwise.
    // conditions is a flat array representing feature_idx, relation (0: <, 1: >), threshold.
    NIZAM_API int evaluate_symbolic_clause(const float* features, int feature_count, const float* conditions, int condition_count);
}

#endif // NIZAM_CORE_H
