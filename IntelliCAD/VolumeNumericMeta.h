#pragma once

#include "Size3D.hpp"

class VolumeNumericMeta
{
public:
	/* constructor */
	VolumeNumericMeta() = default;

	/* member variable */
	Size3D<> memSize = { 0, 0, 0 };
	Size3D<float> spacing = { 1.f, 1.f, 1.f };
	Size3D<float> volSize = { 0, 0, 0 };

	ushort voxelPrecision = USHRT_MAX;
};
