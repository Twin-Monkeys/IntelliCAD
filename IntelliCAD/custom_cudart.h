#pragma once

#ifndef __CUDACC__
#define __CUDACC__
#include <cuda_runtime.h>
#define __host__ 
#define __device__
#define __global__
#define __shared__
#define __constant__
#define __managed__
#undef __CUDACC__
#endif
