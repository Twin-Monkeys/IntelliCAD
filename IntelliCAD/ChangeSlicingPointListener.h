#pragma once

#include "Point3D.h"

class ChangeSlicingPointListener
{
public:
	virtual void onChangeSlicingPoint(const Point3D &slicingPoint) = 0;
};