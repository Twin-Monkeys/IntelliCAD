#pragma once

#include "GPUVolume.h"

class SetVolumeListener
{
public:
	/// <summary>
	/// generic �̺�Ʈ�� ó���ϴ� �Լ��̴�.
	/// </summary>
	virtual void onSetVolume(const GPUVolume *const pVolume) = 0;
};