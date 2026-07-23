import ctypes
import os
import sys
import numpy as np

# Find and load the shared library
lib_path = None
dll_name = "nizam_core.dll" if sys.platform == "win32" else "libnizam_core.so"
possible_paths = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), dll_name),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "core", "build", dll_name),
    os.path.join(os.getcwd(), dll_name),
]

_lib = None
HAS_CPP_CORE = False

for path in possible_paths:
    if os.path.exists(path):
        try:
            _lib = ctypes.CDLL(path)
            HAS_CPP_CORE = True
            lib_path = path
            break
        except Exception as e:
            # Failed to load, will fallback to pure python
            pass

if HAS_CPP_CORE:
    # Setup ctypes argument types and return types
    # void quantize_int8(const float* input, int8_t* output, int size, float scale, int8_t zero_point);
    _lib.quantize_int8.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int8
    ]
    _lib.quantize_int8.restype = None

    # void dequantize_int8(const int8_t* input, float* output, int size, float scale, int8_t zero_point);
    _lib.dequantize_int8.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int8
    ]
    _lib.dequantize_int8.restype = None

    # void quantize_binary(const float* input, int8_t* output, int size);
    _lib.quantize_binary.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int
    ]
    _lib.quantize_binary.restype = None

    # void quantize_int4(const float* input, int8_t* output, int size, float scale, int8_t zero_point);
    _lib.quantize_int4.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int8
    ]
    _lib.quantize_int4.restype = None

    # void dequantize_int4(const int8_t* input, float* output, int size, float scale, int8_t zero_point);
    _lib.dequantize_int4.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int8
    ]
    _lib.dequantize_int4.restype = None

    # void simulate_hardware_vector_ops(const float* a, const float* b, float* result, int size, int hardware_type);
    _lib.simulate_hardware_vector_ops.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.simulate_hardware_vector_ops.restype = None

    # int evaluate_symbolic_clause(const float* features, int feature_count, const float* conditions, int condition_count);
    _lib.evaluate_symbolic_clause.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.evaluate_symbolic_clause.restype = ctypes.c_int


# Pure Python Fallback implementations
def py_quantize_int8(input_array, scale, zero_point):
    if scale == 0.0:
        scale = 1.0
    input_arr = np.array(input_array, dtype=np.float32)
    scaled = (input_arr / scale) + zero_point
    rounded = np.round(scaled)
    clipped = np.clip(rounded, -128, 127)
    return clipped.astype(np.int8)

def py_dequantize_int8(input_array, scale, zero_point):
    input_arr = np.array(input_array, dtype=np.int8)
    dequantized = (input_arr.astype(np.float32) - zero_point) * scale
    return dequantized

def py_quantize_binary(input_array):
    input_arr = np.array(input_array, dtype=np.float32)
    binary = np.where(input_arr >= 0.0, 1, -1)
    return binary.astype(np.int8)

def py_quantize_int4(input_array, scale, zero_point):
    if scale == 0.0:
        scale = 1.0
    input_arr = np.array(input_array, dtype=np.float32)
    scaled = (input_arr / scale) + zero_point
    rounded = np.round(scaled)
    clipped = np.clip(rounded, -8, 7)
    return clipped.astype(np.int8)

def py_dequantize_int4(input_array, scale, zero_point):
    input_arr = np.array(input_array, dtype=np.int8)
    dequantized = (input_arr.astype(np.float32) - zero_point) * scale
    return dequantized

def py_simulate_hardware_vector_ops(a_list, b_list, hardware_type):
    a_arr = np.array(a_list, dtype=np.float32)
    b_arr = np.array(b_list, dtype=np.float32)
    return (a_arr + b_arr).tolist()

def py_evaluate_symbolic_clause(features, conditions):
    # conditions is a list of tuples: (feature_idx, relation, threshold)
    # relation: 0.0 for less than (<), 1.0 for greater than (>)
    for feature_idx, relation, threshold in conditions:
        if feature_idx < 0 or feature_idx >= len(features):
            return 0
        val = features[feature_idx]
        if relation == 0.0:  # <
            if not (val < threshold):
                return 0
        elif relation == 1.0:  # >
            if not (val > threshold):
                return 0
        else:
            if not (abs(val - threshold) < 1e-5):
                return 0
    return 1


# Wrapper functions exposing consistent API
def quantize_int8(float_list, scale, zero_point):
    """
    Quantizes a list of floats into 8-bit integers.
    """
    if HAS_CPP_CORE:
        size = len(float_list)
        c_input = (ctypes.c_float * size)(*float_list)
        c_output = (ctypes.c_int8 * size)()
        _lib.quantize_int8(c_input, c_output, size, scale, zero_point)
        return np.array(c_output, dtype=np.int8)
    else:
        return py_quantize_int8(float_list, scale, zero_point)

def dequantize_int8(int8_list, scale, zero_point):
    """
    Dequantizes a list of 8-bit integers back to floats.
    """
    if HAS_CPP_CORE:
        size = len(int8_list)
        c_input = (ctypes.c_int8 * size)(*int8_list)
        c_output = (ctypes.c_float * size)()
        _lib.dequantize_int8(c_input, c_output, size, scale, zero_point)
        return np.array(c_output, dtype=np.float32)
    else:
        return py_dequantize_int8(int8_list, scale, zero_point)

def quantize_binary(float_list):
    """
    Quantizes a list of floats into 1-bit binary values (-1 or 1).
    """
    if HAS_CPP_CORE:
        size = len(float_list)
        c_input = (ctypes.c_float * size)(*float_list)
        c_output = (ctypes.c_int8 * size)()
        _lib.quantize_binary(c_input, c_output, size)
        return np.array(c_output, dtype=np.int8)
    else:
        return py_quantize_binary(float_list)

def quantize_int4(float_list, scale, zero_point):
    """
    Quantizes a list of floats into 4-bit integers.
    """
    if HAS_CPP_CORE:
        size = len(float_list)
        c_input = (ctypes.c_float * size)(*float_list)
        c_output = (ctypes.c_int8 * size)()
        _lib.quantize_int4(c_input, c_output, size, scale, zero_point)
        return np.array(c_output, dtype=np.int8)
    else:
        return py_quantize_int4(float_list, scale, zero_point)

def dequantize_int4(int8_list, scale, zero_point):
    """
    Dequantizes a list of 4-bit integers back to floats.
    """
    if HAS_CPP_CORE:
        size = len(int8_list)
        c_input = (ctypes.c_int8 * size)(*int8_list)
        c_output = (ctypes.c_float * size)()
        _lib.dequantize_int4(c_input, c_output, size, scale, zero_point)
        return np.array(c_output, dtype=np.float32)
    else:
        return py_dequantize_int4(int8_list, scale, zero_point)

def simulate_hardware_vector_ops(a_list, b_list, hardware_type=0):
    """
    Simulates hardware instruction vector operations for benchmarks.
    """
    if HAS_CPP_CORE:
        size = len(a_list)
        c_a = (ctypes.c_float * size)(*a_list)
        c_b = (ctypes.c_float * size)(*b_list)
        c_res = (ctypes.c_float * size)()
        _lib.simulate_hardware_vector_ops(c_a, c_b, c_res, size, hardware_type)
        return np.array(c_res, dtype=np.float32).tolist()
    else:
        return py_simulate_hardware_vector_ops(a_list, b_list, hardware_type)

def evaluate_symbolic_clause(features, conditions):
    """
    Evaluates list of rules against a feature list.
    conditions is a list of tuples: (feature_idx, relation, threshold)
    where relation is 0.0 (less than) or 1.0 (greater than).
    """
    if HAS_CPP_CORE:
        # Flatten conditions list
        flat_conditions = []
        for cond in conditions:
            flat_conditions.extend([float(cond[0]), float(cond[1]), float(cond[2])])
        
        feature_count = len(features)
        condition_count = len(conditions)
        
        c_features = (ctypes.c_float * feature_count)(*features)
        c_conditions = (ctypes.c_float * len(flat_conditions))(*flat_conditions)
        
        result = _lib.evaluate_symbolic_clause(c_features, feature_count, c_conditions, condition_count)
        return bool(result)
    else:
        return bool(py_evaluate_symbolic_clause(features, conditions))
