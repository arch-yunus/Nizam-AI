#include "nizam_core.h"
#include <cmath>

extern "C" {

    NIZAM_API void quantize_int8(const float* input, int8_t* output, int size, float scale, int8_t zero_point) {
        if (scale == 0.0f) scale = 1.0f;
        for (int i = 0; i < size; ++i) {
            float val = input[i];
            // Scale and shift
            float scaled = val / scale + zero_point;
            // Round to nearest integer
            float rounded = std::round(scaled);
            // Clip to signed 8-bit integer range [-128, 127]
            if (rounded < -128.0f) rounded = -128.0f;
            if (rounded > 127.0f) rounded = 127.0f;
            output[i] = static_cast<int8_t>(rounded);
        }
    }

    NIZAM_API void dequantize_int8(const int8_t* input, float* output, int size, float scale, int8_t zero_point) {
        for (int i = 0; i < size; ++i) {
            output[i] = (static_cast<float>(input[i]) - zero_point) * scale;
        }
    }

    NIZAM_API void quantize_binary(const float* input, int8_t* output, int size) {
        for (int i = 0; i < size; ++i) {
            output[i] = (input[i] >= 0.0f) ? 1 : -1;
        }
    }

    NIZAM_API int evaluate_symbolic_clause(const float* features, int feature_count, const float* conditions, int condition_count) {
        // conditions is a flat array of floats where each condition is 3 elements:
        // [feature_idx, relation, threshold]
        // relation: 0 for less than (<), 1 for greater than (>)
        for (int i = 0; i < condition_count; ++i) {
            int offset = i * 3;
            int feature_idx = static_cast<int>(conditions[offset]);
            float relation = conditions[offset + 1];
            float threshold = conditions[offset + 2];

            if (feature_idx < 0 || feature_idx >= feature_count) {
                return 0; // Index out of bounds, fail check
            }

            float val = features[feature_idx];
            if (relation == 0.0f) { // Less than
                if (!(val < threshold)) {
                    return 0; // Condition violated
                }
            } else if (relation == 1.0f) { // Greater than
                if (!(val > threshold)) {
                    return 0; // Condition violated
                }
            } else { // Equal to (fallback)
                if (std::abs(val - threshold) > 1e-5f) {
                    return 0;
                }
            }
        }
        return 1; // All conditions satisfied
    }
}
