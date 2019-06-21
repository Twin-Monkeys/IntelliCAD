#pragma once

#include <memory>
#include "TypeEx.h"
#include "ColorChannelType.h"

class ColorTransferFunction
{
private:
	float *__pBufferArr[4] = { nullptr, nullptr, nullptr, nullptr };

	void __release();

public:
	const ushort PRECISION;

	ColorTransferFunction(const ushort precision);

	void init(const ColorChannelType colorType);
	const float *getBuffer(const ColorChannelType colorType) const;

	template <typename T>
	void set(const ColorChannelType colorType, const T* const pData);

	template <typename T>
	void set(
		const T* const pDataRed, const T* const pDataGreen,
		const T* const pDataBlue, const T* const pDataAlpha);

	~ColorTransferFunction();
};

template <typename T>
void ColorTransferFunction::set(const ColorChannelType colorType, const T* const pData)
{
	const int ITER = PRECISION;

	if (colorType == ColorChannelType::ALL)
	{
		float *const pBufferRed = __pBufferArr[0];

		for (int i = 0; i < ITER; i++)
			pBufferRed[i] = static_cast<float>(pData[i]);

		const int MEM_SIZE = static_cast<int>(PRECISION * sizeof(float));

		for (int i = 1; i < ITER; i++)
			memcpy(__pBufferArr[i], pBufferRed, MEM_SIZE);

		return;
	}

	float *const pTarget = __pBufferArr[colorType];

	for (int i = 0; i < ITER; i++)
		pTarget[i] = static_cast<float>(pData[i]);
}

template <typename T>
void ColorTransferFunction::set(
	const T* const pDataRed, const T* const pDataGreen, const T* const pDataBlue, const T* const pDataAlpha)
{
	const int ITER = PRECISION;

	float *const pBufferRed = __pBufferArr[0];
	float *const pBufferGreen = __pBufferArr[1];
	float *const pBufferBlue = __pBufferArr[2];
	float *const pBufferAlpha = __pBufferArr[3];

	for (int i = 0; i < ITER; i++)
	{
		pBufferRed[i] = static_cast<float>(pDataRed[i]);
		pBufferGreen[i] = static_cast<float>(pDataGreen[i]);
		pBufferBlue[i] = static_cast<float>(pDataBlue[i]);
		pBufferAlpha[i] = static_cast<float>(pDataAlpha[i]);
	}
}