#include "MonoTransferFunction.h"

MonoTransferFunction::MonoTransferFunction(const ushort precision) :
	PRECISION(precision)
{
	__pBuffer = new float[PRECISION];
}

void MonoTransferFunction::__release()
{
	if (__pBuffer)
	{
		delete[] __pBuffer;
		__pBuffer = nullptr;
	}
}

void MonoTransferFunction::init()
{
	const float RANGE_INV = (1.f / static_cast<float>(PRECISION));
	const int ITER = PRECISION;

	for (int i = 0; i < ITER; i++)
		__pBuffer[i] = (static_cast<float>(i) * RANGE_INV);
}


const float *MonoTransferFunction::getBuffer() const
{
	return __pBuffer;
}

MonoTransferFunction::~MonoTransferFunction()
{
	__release();
}