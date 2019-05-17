#pragma once

#include "VolumeData.h"

class VolumeLoadingListener
{
public:
	virtual void onLoadVolume(const VolumeData &volumeData) = 0;
};