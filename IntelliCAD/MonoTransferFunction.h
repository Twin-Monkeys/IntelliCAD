#pragma once

#include "TypeEx.h"

class MonoTransferFunction
{
private:
	float *__pBuffer = nullptr;

	void __release();

public:
	const ushort PRECISION;

	MonoTransferFunction(const ushort precision);

	void init();
	const float *getBuffer() const;

	template <typename T>
	void set(const T* const pData);

	~MonoTransferFunction();
};

template <typename T>
void MonoTransferFunction::set(const T* const pData)
{
	const int ITER = PRECISION;

	for (int i = 0; i < ITER; i++)
		__pBuffer[i] = static_cast<float>(pData[i]);
}