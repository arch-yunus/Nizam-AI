cmake_minimum_required(VERSION 3.10)
project(NizamCore VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure output directory for the shared library
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Include headers path
include_directories(core/include)

# Define the shared library target
add_library(nizam_core SHARED core/src/nizam_core.cpp)

# Install rules (optional)
install(TARGETS nizam_core
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
