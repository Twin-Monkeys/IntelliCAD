#include "GPUVolume.h"

GPUVolume GPUVolume::__instance;

namespace Device
{
	// static texture<ubyte, 3, cudaReadModeNormalizedFloat> __ubyteTex;
	// static texture<ushort, 3, cudaReadModeNormalizedFloat> __ushortTex;
	// static texture<uint, 3, cudaReadModeNormalizedFloat> __uintTex;
}

#define CUR_TEX \
	((__voxelFormat == 0) ? Device::__ubyteTex : \
	((__voxelFormat == 1) ? Device::__ushortTex : Device::__uintTex))

void GPUVolume::init(
	const ubyte *const pHostSrc,
	const int width, const int height, const int depth,
	const VoxelFormatType voxelFormat, const int voxelPrecision)
{
	__voxelFormat = voxelFormat;
	__volumeSize.set(width, height, depth);
	
	// some initializing logic with cuda texture..
}

const Size3D<>& GPUVolume::getSize() const
{
	return __volumeSize;
}

VoxelFormatType GPUVolume::getVoxelFormat() const
{
	return __voxelFormat;
}

int GPUVolume::getVoxelPrecision() const
{
	return __voxelPrecision;
}

float GPUVolume::get(const float x, const float y, const float z)
{
	// tex3D(CUR_TEX, x, y, z);

	return 0;
}

float GPUVolume::get(const Point3D &position)
{
	// tex3D(CUR_TEX, position.x, position.y, position.z);

	return 0;
}

GPUVolume& GPUVolume::getInstance()
{
	return __instance;
}

GPUVolume::~GPUVolume()
{
	// some release logic with cuda resources
}