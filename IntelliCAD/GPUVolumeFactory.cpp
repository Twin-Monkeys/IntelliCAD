#include "GPUVolumeFactory.h"
#include "VolumeReader.hpp"

using namespace std;

namespace GPUVolumeFactory
{
	GPUVolume *getVolumeWithDen(
		const std::tstring &path,
		const int width, const int height, const int depth, const VoxelFormatType voxelFormat)
	{
		return getVolumeWithDen(path, width, height, depth, voxelFormat, 1 << (8 << voxelFormat));
	}

	GPUVolume *getVolumeWithDen(
		const std::tstring &path, const int width, const int height, const int depth,
		const VoxelFormatType voxelFormat, const int voxelPrecision)
	{
		GPUVolume &instance = GPUVolume::getInstance();

		if (!VolumeReader::readDen(path, instance, width, height, depth, voxelFormat, voxelPrecision))
			return nullptr;

		return &instance;
	}
}