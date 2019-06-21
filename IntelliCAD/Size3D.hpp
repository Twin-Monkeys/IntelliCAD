#pragma once

#include <type_traits>
#include <cuda_runtime.h>
#include "TypeEx.h"

template <typename T = int>
class Size3D
{
	static_assert(std::is_arithmetic_v<T>, "T must be arithmetical type");

public:
	T width;
	T height;
	T depth;

	Size3D() = default;

	__host__ __device__
	constexpr Size3D(T width, T height, T depth);

	__host__ __device__
	void set(T width, T height, T depth);

	template <typename T2 = ubyte>
	__host__ __device__
	T getTotalSize() const;

	template <typename T2>
	__host__ __device__
	Size3D<T2> castTo() const;

	__host__ __device__
	Size3D<T> operator*(const T ratio) const;

	__host__ __device__
	Size3D<T> operator/(const T ratio) const;
};

template <typename T>
__host__ __device__
constexpr Size3D<T>::Size3D(const T width, const T height, const T depth)
	: width(width), height(height), depth(depth)
{}

template <typename T>
__host__ __device__
void Size3D<T>::set(const T width, const T height, const T depth)
{
	this->width = width;
	this->height = height;
	this->depth = depth;
}

template <typename T>
template <typename T2>
__host__ __device__
T Size3D<T>::getTotalSize() const
{
	return (width * height * depth * sizeof(T2));
}

template <typename T>
template <typename T2>
__host__ __device__
Size3D<T2> Size3D<T>::castTo() const
{
	return {
		static_cast<T2>(width),
		static_cast<T2>(height),
		static_cast<T2>(depth)
	};
}

template <typename T>
__host__ __device__
Size3D<T> Size3D<T>::operator*(const T ratio) const
{
	return
	{
		width * ratio,
		height * ratio,
		depth * ratio
	};
}

template <typename T>
__host__ __device__
Size3D<T> Size3D<T>::operator/(const T ratio) const
{
	return
	{
		width / ratio,
		height / ratio,
		depth / ratio
	};
}