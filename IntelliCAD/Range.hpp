#pragma once

#include <cuda_runtime.h>
#include <type_traits>

template <typename T>
class Range
{
	static_assert(std::is_arithmetic_v<T>, "T must be arithmetical type");

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

	template <typename T2>
	__host__ __device__
	Range<T2> castTo() const;
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

template <typename T>
template <typename T2>
__host__ __device__
Range<T2> Range<T>::castTo() const
{
	return {
		static_cast<T>(start),
		static_cast<T>(end)
	};
}