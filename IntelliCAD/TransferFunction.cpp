#include "TransferFunction.h"

using namespace std;

/* constructor */
TransferFunction::TransferFunction(const ushort precision) :
	PRECISION(precision)
{
	__malloc();
}

/* destructor */
TransferFunction::~TransferFunction() 
{
	if (__pRed)
		__free(__pRed);

	if (__pGreen)
		__free(__pGreen);

	if (__pBlue)
		__free(__pBlue);

	if (__pAlpha)
		__free(__pAlpha);
}

/* member function */
void TransferFunction::setRed(const Range<ushort>& filter) 
{
	__calcTransferFunc(__pRed, filter);
}

void TransferFunction::setGreen(const Range<ushort>& filter) 
{
	__calcTransferFunc(__pGreen, filter);
}

void TransferFunction::setBlue(const Range<ushort>& filter) 
{
	__calcTransferFunc(__pBlue, filter);
}

void TransferFunction::setAlpha(const Range<ushort>& filter) 
{
	__calcTransferFunc(__pAlpha, filter);
}

const float* TransferFunction::getRed() const 
{
	return const_cast<const float*>(__pRed);
}

const float* TransferFunction::getGreen() const 
{
	return const_cast<const float*>(__pGreen);
}

const float* TransferFunction::getBlue() const 
{
	return const_cast<const float*>(__pBlue);
}

const float* TransferFunction::getAlpha() const 
{
	return const_cast<const float*>(__pAlpha);
}

void TransferFunction::__malloc() 
{
	__pRed = new float[PRECISION];
	__pGreen = new float[PRECISION];
	__pBlue = new float[PRECISION];
	__pAlpha = new float[PRECISION];
}
	
void TransferFunction::__calcTransferFunc(float* const pTransferFunc, const Range<ushort>& filter) 
{
	const float INV_RANGE = (1.f / static_cast<float>(filter.getGap()));

	for (int i = 0; i < PRECISION; ++i)
	{
		if (i < filter.start)
			pTransferFunc[i] = 0.f;
		else if (i < filter.end)
			pTransferFunc[i] = (static_cast<float>(i - filter.start) * INV_RANGE);
		else
			pTransferFunc[i] = 1.f;
	}
}
	
void TransferFunction::__free(float*& pTransferFunc) 
{
	delete[] pTransferFunc;
	pTransferFunc = nullptr;
}