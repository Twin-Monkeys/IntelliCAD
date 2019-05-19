#include <iostream>
#include "CudaHelper.h"

using namespace std;

void CudaHelper::optimizeLaunchingParams(const dim3 &totalDim, dim3 &gridDim, dim3 &blockDim, const uint idealBlockSize)
{

	const uint TOTAL_DIM_ARR[] = {
		totalDim.x, totalDim.y, totalDim.z
	};

	uint gridDimArr[] = {
		totalDim.x, totalDim.y, totalDim.z
	};

	uint blockDimArr[] = { 1, 1, 1 };

	int arrIdx = 0;
	do
	{
		uint tmpBlockDimArr[] = {
			blockDimArr[0], blockDimArr[1], blockDimArr[2]
		};

		bool outerLooping = true;
		bool innerLooping = true;
		do
		{
			for (int i = 0; i < 3; i++)
			{
				arrIdx = ((arrIdx + 1) % 3);
				if (!(TOTAL_DIM_ARR[arrIdx] % ++tmpBlockDimArr[arrIdx]))
				{
					innerLooping = false;
					break;
				}
			}

			if (
				(TOTAL_DIM_ARR[0] <= tmpBlockDimArr[0]) &&
				(TOTAL_DIM_ARR[1] <= tmpBlockDimArr[1]) &&
				(TOTAL_DIM_ARR[2] <= tmpBlockDimArr[2]))
			{
				outerLooping = false;
				break;
			}
		}
		while (innerLooping);

		if (!outerLooping)
			break;

		blockDimArr[arrIdx] = tmpBlockDimArr[arrIdx];
		gridDimArr[arrIdx] = (TOTAL_DIM_ARR[arrIdx] / blockDimArr[arrIdx]);

		const uint BLOCK_SIZE = (blockDimArr[0] * blockDimArr[1] * blockDimArr[2]);

		if (BLOCK_SIZE >= idealBlockSize)
			break;
	}
	while (true);

	gridDim = {
		gridDimArr[0], gridDimArr[1], gridDimArr[2]
	};

	blockDim = {
		blockDimArr[0], blockDimArr[1], blockDimArr[2]
	};
}

void CudaHelper::optimizeLaunchingParams(const Size2D<> &totalDim, dim3 &gridDim, dim3 &blockDim, const uint idealBlockSize)
{
	optimizeLaunchingParams(dim3(totalDim.width, totalDim.height), gridDim, blockDim, idealBlockSize);
}

void CudaHelper::optimizeLaunchingParams(const Size3D<> &totalDim, dim3 &gridDim, dim3 &blockDim, const uint idealBlockSize)
{
	optimizeLaunchingParams(dim3(totalDim.width, totalDim.height, totalDim.depth), gridDim, blockDim, idealBlockSize);
}