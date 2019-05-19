#pragma once

#include "Size3D.hpp"

class VolumeMeta
{
public:
	/* constructor */
	VolumeMeta() = default;

	/* member variable */
	Size3D<> size = { 0, 0, 0 };
	Size3D<float> spacing = { 1.f, 1.f, 1.f };
	ushort voxelPrecision = USHRT_MAX;
};
