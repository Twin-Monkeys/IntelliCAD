#pragma once

#include "Size3D.hpp"
#include "Point3D.h"

namespace VolumeIndicator
{
	__device__
	bool recognize(const Size3D<float> &volSize, const Point3D &target, const float lineLength, const float lineThickness);
}
