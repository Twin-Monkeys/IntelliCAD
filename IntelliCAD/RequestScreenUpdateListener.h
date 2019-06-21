#pragma once

#include "RenderingScreenType.h"

class RequestScreenUpdateListener
{
public:
	virtual void onRequestScreenUpdate(const RenderingScreenType targetType) = 0;
};
