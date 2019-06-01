#pragma once

#include <type_traits>
#include <cuda_runtime.h>
#include "TypeEx.h"

template <typename T = int>
class Size2D
{
	static_assert(std::is_arithmetic_v<T>, "T must be arithmetical type");

public:
	T width;
	T height;

	Size2D() = default;

	__host__ __device__
	constexpr Size2D(T width, T height);

	__host__ __device__
	void set(T width, T height);

	template <typename T2 = ubyte>
	__host__ __device__
	T getTotalSize() const;

	template <typename T2>
	__host__ __device__
	Size2D<T2> castTo() const;

	__host__ __device__
	Size2D<T> operator/(const T divider) const;
};

template <typename T>
__host__ __device__
constexpr Size2D<T>::Size2D(const T width, const T height)
	: width(width), height(height)
{}

template <typename T>
__host__ __device__
void Size2D<T>::set(const T width, const T height)
{
	this->width = width;
	this->height = height;
}

template <typename T>
template <typename T2>
__host__ __device__
T Size2D<T>::getTotalSize() const
{
	return (width * height * sizeof(T2));
}

template <typename T>
template <typename T2>
__host__ __device__
Size2D<T2> Size2D<T>::castTo() const
{
	return {
		static_cast<T2>(width),
		static_cast<T2>(height)
	};
}

template <typename T>
__host__ __device__
Size2D<T> Size2D<T>::operator/(const T divider) const
{
	return
	{
		width / divider,
		height / divider
	};
}