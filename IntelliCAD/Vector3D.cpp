#include <cmath>
#include "Vector3D.h"

const Vector3D Vector3D::AXIS_X = { 1.f, 0.f, 0.f };
const Vector3D Vector3D::AXIS_Y = { 0.f, 1.f, 0.f };
const Vector3D Vector3D::AXIS_Z = { 0.f, 0.f, -1.f };

__host__ __device__
Vector3D::Vector3D(const float x, const float y, const float z)
	: x(x), y(y), z(z)
{}

__host__ __device__
Vector3D::Vector3D(const float value)
	: x(value), y(value), z(value)
{}

__host__ __device__
void Vector3D::set(const float x, const float y, const float z)
{
	this->x = x;
	this->y = y;
	this->z = z;
}

__host__ __device__
void Vector3D::set(const Vector3D &another)
{
	x = another.x;
	y = another.y;
	z = another.z;
}

__host__ __device__
void Vector3D::normalize()
{
	(*this /= getLength());
}

__host__ __device__
float Vector3D::dot(const float x, const float y, const float z) const
{
	return ((this->x * x) + (this->y * y) + (this->z * z));
}

__host__ __device__
float Vector3D::dot(const Vector3D &another) const
{
	return dot(another.x, another.y, another.z);
}

__host__ __device__
Vector3D Vector3D::cross(const float x, const float y, const float z) const
{
	const float xVal = ((this->y * z) - (this->z * y));
	const float yVal = ((this->z * x) - (this->x * z));
	const float zVal = ((this->x * y) - (this->y * x));

	return { xVal, yVal, zVal };
}

__host__ __device__
Vector3D Vector3D::cross(const Vector3D &another) const
{
	return cross(another.x, another.y, another.z);
}

__host__ __device__
Vector3D Vector3D::rotate(const Vector3D &axis, const float angle) const
{
	const float s = sinf(angle);
	const float c = cosf(angle);

	const Vector3D matCol1 = {
		((1 - c) * (axis.x * axis.x)) + c,
		((1 - c) * (axis.y * axis.x)) - (s * axis.z),
		((1 - c) * (axis.z * axis.x)) + (s * axis.y)
	};

	const Vector3D matCol2 = {
		((1 - c) * (axis.x * axis.y)) + (s * axis.z),
		((1 - c) * (axis.y * axis.y)) + c,
		((1 - c) * (axis.z * axis.y)) - (s * axis.x)
	};

	const Vector3D matCol3 = {
		((1 - c) * (axis.x * axis.z)) - (s * axis.y),
		((1 - c) * (axis.y * axis.z)) + (s * axis.x),
		((1 - c) * (axis.z * axis.z)) + c
	};

	return {
		this->dot(matCol1),
		this->dot(matCol2),
		this->dot(matCol3)
	};
}

__host__ __device__
float Vector3D::getLengthSq() const
{
	return ((x * x) + (y * y) + (z * z));
}

__host__ __device__
float Vector3D::getLength() const
{
	return sqrt(getLengthSq());
}

__host__ __device__
Vector3D Vector3D::getUnit() const
{
	return (*this / getLength());
}

__host__ __device__
Vector3D Vector3D::operator+(const Vector3D &another) const
{
	return Vector3D(x + another.x, y + another.y, z + another.z);
}

__host__ __device__
Vector3D& Vector3D::operator+=(const Vector3D &another)
{
	x += another.x;
	y += another.y;
	z += another.z;

	return *this;
}

__host__ __device__
Vector3D Vector3D::operator-(const Vector3D &another) const
{
	return {
		x - another.x,
		y - another.y,
		z - another.z
	};
}

__host__ __device__
Vector3D Vector3D::operator-() const
{
	return { -x, -y, -z };
}

__host__ __device__
Vector3D& Vector3D::operator-=(const Vector3D &another)
{
	x -= another.x;
	y -= another.y;
	z -= another.z;

	return *this;
}

__host__ __device__
Vector3D Vector3D::operator*(const float ratio) const
{
	return Vector3D(x * ratio, y * ratio, z * ratio);
}

__host__ __device__
Vector3D& Vector3D::operator*=(const float ratio)
{
	x *= ratio;
	y *= ratio;
	z *= ratio;

	return *this;
}

__host__ __device__
Vector3D Vector3D::operator/(const float ratio) const
{
	const float ratioInv = (1.f / ratio);

	return Vector3D(x * ratioInv, y * ratioInv, z * ratioInv);
}

__host__ __device__
Vector3D& Vector3D::operator/=(const float ratio)
{
	const float ratioInv = (1.f / ratio);

	x *= ratioInv;
	y *= ratioInv;
	z *= ratioInv;

	return *this;
}

__host__ __device__
Vector3D operator*(const float ratio, const Vector3D &vector)
{
	return (vector * ratio);
}