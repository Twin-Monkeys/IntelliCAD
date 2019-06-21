#pragma once

#include "ColorChannelType.h"

class InitVolumeTransferFunctionListener
{
public:
	virtual void onInitVolumeTransferFunction(const ColorChannelType colorType) = 0;
};