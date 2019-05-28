#pragma once

#include "TypeEx.h"
#include "Size2D.hpp"
#include "Size3D.hpp"

class CudaHelper
{
private:
	CudaHelper() = delete;

public:
	static void optimizeLaunchingParams(const dim3 &totalDim, dim3 &gridDim, dim3 &blockDim, uint idealBlockSize = 256U);
	static void optimizeLaunchingParams(const Size2D<> &totalDim, dim3 &gridDim, dim3 &blockDim, uint idealBlockSize = 256U);
	static void optimizeLaunchingParams(const Size3D<> &totalDim, dim3 &gridDim, dim3 &blockDim, uint idealBlockSize = 256U);
};
