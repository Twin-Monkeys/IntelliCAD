#pragma once

#include "SliceAxis.h"
#include "Point2D.h"

class InitSliceViewAnchorListener
{
public:
	virtual void onInitSliceViewAnchor(const SliceAxis axis) = 0;
};