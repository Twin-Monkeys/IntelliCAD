/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	���ϸ�			: Color.hpp
*	�ۼ���			: ����
*	���� ������		: 19.03.05
*/

#pragma once

#include <cuda_runtime.h>

template <typename T>
class Color
{
public:
	/* constructor */
	/// <summary>
	/// ������
	/// </summary>
	Color() = default;

	/// <summary>
	/// ������
	/// </summary>
	/// <param name="intensity">
	/// red, green, blue�� �ʱ�ȭ�� ��
	/// </param>
	__host__ __device__
	explicit Color(const T intensity);

	/// <summary>
	/// ������
	/// </summary>
	/// <param name="red">
	/// red ���� �ʱ�ȭ �Ѵ�
	/// </param>
	/// <param name="green">
	/// green ���� �ʱ�ȭ �Ѵ�
	/// </param>
	/// <param name="blue">
	///	blue ���� �ʱ�ȭ �Ѵ�
	/// </param>
	__host__ __device__
	Color(const T red, const T green, const T blue);

	/* member function */
	/// <summary>
	/// red, green, blue ���� �����Ѵ�
	/// </summary>
	/// <param name="intensity">
	/// red, green, blue�� ������ ��
	/// </param>
	__host__ __device__
	void set(const T intensity);

	/// <summary>
	/// red, green, blue ���� �����Ѵ� 
	/// </summary>
	/// <param name="red">
	/// red ���� �����Ѵ�
	/// </param>
	/// <param name="green">
	/// green ���� �����Ѵ�
	/// </param>
	/// <param name="blue">
	/// blue ���� �����Ѵ�
	/// </param>
	__host__ __device__
	void set(const T red, const T green, const T blue);

	/// <summary>
	/// �� �÷��� �����ϴ� ���� �� ��Į�� ���� �����Ѵ�.
	/// </summary>
	/// <param name="operand">
	/// ��Į�� ���� ������ �÷�
	/// </param>
	/// <returns>
	/// ��Į�� �� ���
	/// </returns>
	__host__ __device__
	Color<T> operator*(const Color<T>& operand) const;

	/// <summary>
	/// ��Į�� ���� �����Ѵ�.
	/// </summary>
	/// <param name="value">
	/// ��Į�� ���� ������ ��
	/// </param>
	/// <returns>
	/// ��Į�� �� ���
	/// </returns>
	__host__ __device__
	Color<T> operator*(const T value) const;

	/// <summary>
	/// ���� ����ŭ ������ �����Ѵ�
	/// </summary>
	/// <param name="operand">
	/// ������ ������ �÷�
	/// </param>
	/// <returns>
	/// ���� ��ü�� ���۷���
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