#pragma once

#include "VolumeMeta.h"

class VolumeLoadedListener
{
public:
	virtual void onVolumeLoaded(const VolumeMeta &volumeMeta) = 0;
};