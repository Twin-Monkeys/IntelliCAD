#pragma once

#include <cuda_runtime.h>

template <typename T>
class Range
{
public:
	T start;
	T end;

	Range() = default;

	__host__ __device__
	Range(T start, T end);

	__host__ __device__
	void set(T start, T end);

	__host__ __device__
	T getGap() const;
};

template <typename T>
__host__ __device__
Range<T>::Range(const T start, const T end)
	: start(start), end(end)
{}

template <typename T>
__host__ __device__
void Range<T>::set(const T start, const T end)
{
	this->start = start;
	this->end = end;
}

template <typename T>
__host__ __device__
T Range<T>::getGap() const
{
	return (end - start);
}