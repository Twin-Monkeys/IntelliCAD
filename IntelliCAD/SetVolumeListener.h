#pragma once

#include "GPUVolume.h"

class SetVolumeListener
{
public:
	/// <summary>
	/// generic 이벤트를 처리하는 함수이다.
	/// </summary>
	virtual void onSetVolume(const GPUVolume *const pVolume) = 0;
};