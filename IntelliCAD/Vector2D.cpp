/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	파일명			: Vector2D.cpp
*	작성자			: 원진, 이세인
*	최종 수정일		: 19.05.30
*/

#include <cmath>
#include "Vector2D.h"
#include "NumberUtility.hpp"

__host__ __device__
Vector2D::Vector2D(const float x, const float y) :
	x(x), y(y)
{}

__host__ __device__
Vector2D::Vector2D(const float value) :
	x(value), y(value)
{}

__host__ __device__
void Vector2D::set(const float x, const float y)
{
	this->x = x;
	this->y = y;
}

__host__ __device__
void Vector2D::set(const Vector2D &another)
{
	x = another.x;
	y = another.y;
}

__host__ __device__
void Vector2D::normalize()
{
	(*this /= getLength());
}

__host__ __device__
float Vector2D::dot(const float x, const float y) const
{
	return ((this->x * x) + (this->y * y));
}

__host__ __device__
float Vector2D::dot(const Vector2D &another) const
{
	return dot(another.x, another.y);
}

__host__ __device__
float Vector2D::getLengthSq() const
{
	return ((x * x) + (y * y));
}

__host__ __device__
float Vector2D::getLength() const
{
	return sqrt(getLengthSq());
}

__host__ __device__
Vector2D Vector2D::getUnit() const
{
	return (*this / getLength());
}

__host__ __device__
bool Vector2D::isZero() const
{
	return (*this == Vector2D(0.f, 0.f));
}

__host__ __device__
Vector2D Vector2D::operator+(const Vector2D &another) const
{
	return
	{
		x + another.x,
		y + another.y
	};
}

__host__ __device__
Vector2D& Vector2D::operator+=(const Vector2D &another)
{
	x += another.x;
	y += another.y;

	return *this;
}

__host__ __device__
Vector2D Vector2D::operator-(const Vector2D &another) const
{
	return
	{
		x - another.x,
		y - another.y
	};
}

__host__ __device__
Vector2D Vector2D::operator-() const
{
	return { -x, -y };
}

__host__ __device__
Vector2D& Vector2D::operator-=(const Vector2D &another)
{
	x -= another.x;
	y -= another.y;

	return *this;
}

__host__ __device__
Vector2D Vector2D::operator*(const float ratio) const
{
	return { x * ratio, y * ratio };
}

__host__ __device__
Vector2D& Vector2D::operator*=(const float ratio)
{
	x *= ratio;
	y *= ratio;

	return *this;
}

__host__ __device__
Vector2D Vector2D::operator/(const float ratio) const
{
	const float ratioInv = (1.f / ratio);

	return { x * ratioInv, y * ratioInv };
}

__host__ __device__
Vector2D& Vector2D::operator/=(const float ratio)
{
	const float ratioInv = (1.f / ratio);

	x *= ratioInv;
	y *= ratioInv;

	return *this;
}

__host__ __device__
bool Vector2D::operator==(const Vector2D& operand) const
{
	return (
		NumberUtility::nearEqual(x, operand.x) &&
		NumberUtility::nearEqual(y, operand.y));
}

__host__ __device__
bool Vector2D::operator!=(const Vector2D& operand) const
{
	return !(*this == operand);
}

__host__ __device__
Vector2D operator*(const float ratio, const Vector2D &vector)
{
	return (vector * ratio);
}