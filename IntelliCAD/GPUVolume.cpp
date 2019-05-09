#include "GPUVolume.h"

GPUVolume::GPUVolume(const int width, const int height, const int depth, const int elementSize) :
	SIZE(width, height, depth), ELEMENT_SIZE(elementSize), ELEMENT_PRECISION(1 << (elementSize * 8))
{

}

GPUVolume::GPUVolume(
	const int width, const int height, const int depth, const int elementSize, const int elementPrecision) :
	SIZE(width, height, depth), ELEMENT_SIZE(elementSize), ELEMENT_PRECISION(elementPrecision)
{

}

GPUVolume::GPUVolume(const Size3D<> &size, const int elementSize) :
	SIZE(size), ELEMENT_SIZE(elementSize), ELEMENT_PRECISION(1 << (elementSize * 8))
{

}

GPUVolume::GPUVolume(const Size3D<> &size, const int elementSize, const int elementPrecision) :
	SIZE(size), ELEMENT_SIZE(elementSize), ELEMENT_PRECISION(1 << elementPrecision)
{

}

void GPUVolume::zeroInit()
{

}

GPUVolume::~GPUVolume()
{

}