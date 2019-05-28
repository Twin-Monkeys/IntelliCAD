#pragma once

#include <memory>
#include "TypeEx.h"
#include "tstring.h"
#include "VolumeMeta.h"

class VolumeData
{
public:
	std::shared_ptr<ushort[]> pBuffer = nullptr;

	std::tstring fileName;
	VolumeMeta meta;
};
