#pragma once

#include "Point3D.h"

class InitSlicingPointListener
{
public:
	virtual void onInitSlicingPoint(const Point3D &slicingPoint) = 0;
};