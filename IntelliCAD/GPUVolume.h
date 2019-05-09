#pragma once

#include "Size3D.hpp"

class GPUVolume
{
private:
	ubyte *pDevBuffer = nullptr;

public:
	const Size3D<> SIZE;
	const int ELEMENT_SIZE;
	const int ELEMENT_PRECISION;
	
	GPUVolume(const int width, const int height, const int depth, const int elementSize);
	GPUVolume(const int width, const int height, const int depth, const int elementSize, const int elementPrecision);

	GPUVolume(const Size3D<> &size, const int elementSize);
	GPUVolume(const Size3D<> &size, const int elementSize, const int elementPrecision);

	void zeroInit();

	template <typename T>
	T *getDevBuffer() const;

	~GPUVolume();
};

template <typename T>
T *GPUVolume::getDevBuffer() const
{

}