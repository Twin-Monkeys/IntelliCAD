/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	파일명			: Color.hpp
*	작성자			: 원진
*	최종 수정일		: 19.03.05
*/

#pragma once

#include <cuda_runtime.h>

template <typename T>
class Color
{
public:
	/* constructor */
	/// <summary>
	/// 생성자
	/// </summary>
	Color() = default;

	/// <summary>
	/// 생성자
	/// </summary>
	/// <param name="intensity">
	/// red, green, blue를 초기화할 값
	/// </param>
	__host__ __device__
	explicit Color(const T intensity);

	/// <summary>
	/// 생성자
	/// </summary>
	/// <param name="red">
	/// red 값을 초기화 한다
	/// </param>
	/// <param name="green">
	/// green 값을 초기화 한다
	/// </param>
	/// <param name="blue">
	///	blue 값을 초기화 한다
	/// </param>
	__host__ __device__
	Color(const T red, const T green, const T blue);

	/* member function */
	/// <summary>
	/// red, green, blue 값을 설정한다
	/// </summary>
	/// <param name="intensity">
	/// red, green, blue에 설정할 값
	/// </param>
	__host__ __device__
	void set(const T intensity);

	/// <summary>
	/// red, green, blue 값을 설정한다 
	/// </summary>
	/// <param name="red">
	/// red 값을 설정한다
	/// </param>
	/// <param name="green">
	/// green 값을 설정한다
	/// </param>
	/// <param name="blue">
	/// blue 값을 설정한다
	/// </param>
	__host__ __device__
	void set(const T red, const T green, const T blue);

	/// <summary>
	/// 두 컬러의 대응하는 원소 간 스칼라 곱을 수행한다.
	/// </summary>
	/// <param name="operand">
	/// 스칼라 곱을 수행할 컬러
	/// </param>
	/// <returns>
	/// 스칼라 곱 결과
	/// </returns>
	__host__ __device__
	Color<T> operator*(const Color<T>& operand) const;

	/// <summary>
	/// 스칼라 곱을 수행한다.
	/// </summary>
	/// <param name="value">
	/// 스칼라 곱을 수행할 값
	/// </param>
	/// <returns>
	/// 스칼라 곱 결과
	/// </returns>
	__host__ __device__
	Color<T> operator*(const T value) const;

	/// <summary>
	/// 인자 값만큼 덧셈을 수행한다
	/// </summary>
	/// <param name="operand">
	/// 덧셈을 수행할 컬러
	/// </param>
	/// <returns>
	/// 현재 객체의 레퍼런스
	/// </returns>
	__host__ __device__
	Color<T>& operator+=(const Color<T>& operand);

	/* member variable */
	T red;
	T green;
	T blue;
};

template <typename T>
__host__ __device__ 
Color<T>::Color(const T intensity) :
	red(intensity), green(intensity), blue(intensity)
{}

template <typename T>
__host__ __device__ 
Color<T>::Color(const T red, const T green, const T blue) :
	red(red), green(green), blue(blue)
{}

template <typename T>
__host__ __device__ 
void Color<T>::set(const T intensity)
{
	red = intensity;
	green = intensity;
	blue = intensity;
}

template <typename T>
__host__ __device__ 
void Color<T>::set(const T red, const T green, const T blue)
{
	this->red = red;
	this->green = green;
	this->blue = blue;
}

template <typename T>
__host__ __device__ 
Color<T> Color<T>::operator*(const Color<T>& operand) const
{
	const T RED = (red * operand.red);
	const T GREEN = (green * operand.green);
	const T BLUE = (blue * operand.blue);

	return Color<T>(RED, GREEN, BLUE);
}

template <typename T>
__host__ __device__ 
Color<T> Color<T>::operator*(const T value) const
{
	const T RED = (red * value);
	const T GREEN = (green * value);
	const T BLUE = (blue * value);

	return Color<T>(RED, GREEN, BLUE);
}

template <typename T>
__host__ __device__
Color<T>& Color<T>::operator+=(const Color<T>& operand)
{
	red += operand.red;
	green += operand.green;
	blue += operand.blue;

	return *this;
}