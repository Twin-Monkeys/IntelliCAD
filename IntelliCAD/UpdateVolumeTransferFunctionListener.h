#pragma once

#include "ColorChannelType.h"

class UpdateVolumeTransferFunctionListener
{
public:
	virtual void onUpdateVolumeTransferFunction(const ColorChannelType colorType) = 0;
};