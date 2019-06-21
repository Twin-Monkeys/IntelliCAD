#pragma once

#include "SliceAxis.h"

class UpdateAnchorFromViewListener
{
public:
	virtual void onUpdateAnchorFromView(const SliceAxis axis) = 0;
};