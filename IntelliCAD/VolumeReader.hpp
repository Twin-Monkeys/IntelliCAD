#pragma once

#include <fstream>
#include "tstring.h"
#include "GPUVolume.h"
#include "VoxelFormatType.h"

namespace VolumeReader
{
	bool readDen(
		const std::tstring &path, GPUVolume &volume,
		const int width, const int height, const int depth,
		const VoxelFormatType voxelFormat, const int voxelPrecision);
}
