#pragma once

#include "ColorTransferFunction.h"

ColorTransferFunction::ColorTransferFunction(const ushort precision) :
	PRECISION(precision)
{
	for (int i = 0; i < 4; i++)
		__pBufferArr[i] = new float[PRECISION];
}

void ColorTransferFunction::__release()
{
	for (int i = 0; i < 4; i++)
	{
		if (__pBufferArr[i])
		{
			delete[] __pBufferArr[i];
			__pBufferArr[i] = nullptr;
		}
	}
}

void ColorTransferFunction::init(const ColorChannelType colorType)
{
	const float RANGE_INV = (1.f / static_cast<float>(PRECISION));
	const int ITER = PRECISION;

	if (colorType == ColorChannelType::ALL)
	{
		float *const pRedBuffer = __pBufferArr[0];

		for (int i = 0; i < ITER; i++)
			pRedBuffer[i] = (static_cast<float>(i) * RANGE_INV);

		const int MEM_SIZE = static_cast<int>(PRECISION * sizeof(float));

		for (int i = 1; i < 4; i++)
			memcpy(__pBufferArr[i], pRedBuffer, MEM_SIZE);

		return;
	}

	float *const pBuffer = __pBufferArr[colorType];

	for (int i = 0; i < ITER; i++)
		pBuffer[i] = (static_cast<float>(i) * RANGE_INV);
}

const float *ColorTransferFunction::getBuffer(const ColorChannelType colorType) const
{
	return __pBufferArr[colorType];
}

ColorTransferFunction::~ColorTransferFunction()
{
	__release();
}