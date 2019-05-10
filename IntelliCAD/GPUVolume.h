#pragma once

#include "Size3D.hpp"
#include "Point3D.h"
#include "VoxelFormatType.h"

class GPUVolume
{
private:
	static GPUVolume __instance;

	Size3D<> __volumeSize;
	VoxelFormatType __voxelFormat;
	int __voxelPrecision;

public:
	void init(
		const ubyte *const pHostSrc,
		const int width, const int height, const int depth,
		const VoxelFormatType voxelFormat, const int voxelPrecision);

	const Size3D<>& getSize() const;
	VoxelFormatType getVoxelFormat() const;
	int getVoxelPrecision() const;

	float get(const float x, const float y, const float z);
	float get(const Point3D &position);

	static GPUVolume& getInstance();

	~GPUVolume();
};