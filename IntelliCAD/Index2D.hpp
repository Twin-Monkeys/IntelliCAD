#pragma once

#include <type_traits>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "TypeEx.h"

template <typename T = int>
class Index2D
{
	static_assert(std::is_arithmetic_v<T>, "T must be arithmetical type");

public:
	T x;
	T y;

	Index2D() = default;

	__host__ __device__
	constexpr Index2D(T x, T y);

	__host__ __device__
	void set(T x, T y);

	__host__ __device__
	T getIndex1D() const;

	__device__
	static Index2D getKernelIndex();

	__device__
	static T getKernelIndex1D();
};

template <typename T>
__host__ __device__
constexpr Index2D<T>::Index2D(const T x, const T y)
	: x(x), y(y)
{}

template <typename T>
__host__ __device__
void Index2D<T>::set(const T x, const T y)
{
	this->x = x;
	this->y = y;
}

template <typename T>
__host__ __device__
T Index2D<T>::getIndex1D() const
{
	return ((y * gridDim.x * blockDim.x) + x);
}

template <typename T>
__device__
Index2D<T> Index2D<T>::getKernelIndex()
{
	return {
		static_cast<T>((blockIdx.x * blockDim.x) + threadIdx.x),
		static_cast<T>((blockIdx.y * blockDim.y) + threadIdx.y)
	};
}

template <typename T>
__device__
T Index2D<T>::getKernelIndex1D()
{
	const T X = static_cast<T>((blockIdx.x * blockDim.x) + threadIdx.x);
	const T Y = static_cast<T>((blockIdx.y * blockDim.y) + threadIdx.y);

	return ((Y * gridDim.x * blockDim.x) + X);
}