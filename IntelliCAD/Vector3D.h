#pragma once

#include <cuda_runtime.h>

class Vector3D
{
public:
	union
	{
		struct
		{
			float x;
			float y;
			float z;
		};

		float xyz[3];
	};

	static const Vector3D AXIS_X;
	static const Vector3D AXIS_Y;
	static const Vector3D AXIS_Z;

	Vector3D() = default;

	__host__ __device__
	Vector3D(float x, float y, float z);

	__host__ __device__
	Vector3D(float value);

	__host__ __device__
	void set(float x, float y, float z);

	__host__ __device__
	void set(const Vector3D &another);

	__host__ __device__
	void normalize();

	__host__ __device__
	float dot(float x, float y, float z) const;

	__host__ __device__
	float dot(const Vector3D &another) const;

	__host__ __device__
	Vector3D cross(float x, float y, float z) const;

	__host__ __device__
	Vector3D cross(const Vector3D &another) const;

	__host__ __device__
	Vector3D rotate(const Vector3D &axis, float angle) const;

	__host__ __device__
	float getLengthSq() const;

	__host__ __device__
	float getLength() const;

	__host__ __device__
	Vector3D getUnit() const;

	__host__ __device__
	Vector3D operator+(const Vector3D &another) const;
	
	__host__ __device__
	Vector3D& operator+=(const Vector3D &another);
	
	__host__ __device__
	Vector3D operator-(const Vector3D &another) const;
	
	__host__ __device__
	Vector3D operator-() const;
	
	__host__ __device__
	Vector3D& operator-=(const Vector3D &another);
	
	__host__ __device__
	Vector3D operator*(float ratio) const;
	
	__host__ __device__
	Vector3D& operator*=(float ratio);
	
	__host__ __device__
	Vector3D operator/(float ratio) const;
	
	__host__ __device__
	Vector3D& operator/=(float ratio);
};

__host__ __device__
Vector3D operator*(float ratio, const Vector3D &vector);
