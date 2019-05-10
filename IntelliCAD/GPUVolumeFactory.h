#pragma once

#include "GPUVolume.h"
#include "VoxelFormatType.h"
#include "tstring.h"

namespace GPUVolumeFactory
{
	GPUVolume *getVolumeWithDen(
		const std::tstring &path,
		const int width, const int height, const int depth, const VoxelFormatType voxelFormat);

	GPUVolume *getVolumeWithDen(
		const std::tstring &path, const int width, const int height, const int depth,
		const VoxelFormatType voxelFormat, const int voxelPrecision);
}
