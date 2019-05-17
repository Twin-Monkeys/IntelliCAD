#pragma once

#include <memory>
#include "TypeEx.h"
#include "VolumeMeta.h"

class VolumeData
{
public:
	std::shared_ptr<ushort[]> pBuffer = nullptr;
	VolumeMeta meta;
};
