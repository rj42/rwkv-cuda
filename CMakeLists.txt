IF (APPLE)
    cmake_minimum_required(VERSION 3.4)
ELSE()
    cmake_minimum_required(VERSION 2.8)
ENDIF()

project(rwkv_cuda)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(include)

FIND_PACKAGE(CUDA)
MESSAGE(STATUS "cuda found ${CUDA_FOUND}")

option(HAS_CUDA "cuda check." ${CUDA_FOUND})

IF (NOT HAS_CUDA)
    message(FATAL_ERROR "CUDA not found.")
ENDIF()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")
IF (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5)
  SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
ENDIF()

IF (CUDA_VERSION GREATER 7.6)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
ENDIF()

IF (CUDA_VERSION GREATER 8.9)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
ENDIF()

IF (CUDA_VERSION GREATER 9.9)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
ENDIF()

IF (CUDA_VERSION EQUAL 11)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_52 -gencode=arch=compute_80,code=sm_80")
ENDIF()

IF (CUDA_VERSION GREATER 11)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86")
ENDIF()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11 -O3 -Xptxas -O3")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --extra-device-vectorization --use_fast_math")

MESSAGE(STATUS "Building shared library with GPU support")

CUDA_ADD_LIBRARY(rwkv_cuda SHARED src/wkv_cuda.cu)

INSTALL(TARGETS rwkv_cuda
        RUNTIME DESTINATION "bin"
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib")
