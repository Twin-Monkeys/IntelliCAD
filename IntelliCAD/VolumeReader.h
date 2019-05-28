#pragma once

#include "VolumeData.h"
#include "TypeEx.h"
#include "tstring.h"

namespace VolumeReader
{
	VolumeData readDen(
		const std::tstring &path, const int width, const int height, const int depth);

	VolumeData readDen(
		const std::tstring &path,
		const int width, const int height, const int depth, const ushort voxelPrecision);

	VolumeData readMetaImage(const std::tstring &path);
}