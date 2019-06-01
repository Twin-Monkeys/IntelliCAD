/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	���ϸ�			: Vector2D.h
*	�ۼ���			: �̼���
*	���� ������		: 19.05.30
*/

#pragma once

#include <cuda_runtime.h>

class Vector2D
{
public:
	union
	{
		struct
		{
			float x;
			float y;
		};

		float xy[2];
	};

	Vector2D() = default;

	__host__ __device__
	Vector2D(float x, float y);

	__host__ __device__
	Vector2D(float value);

	__host__ __device__
	void set(float x, float y);

	__host__ __device__
	void set(const Vector2D &another);

	__host__ __device__
	void normalize();

	__host__ __device__
	float dot(float x, float y) const;

	__host__ __device__
	float dot(const Vector2D &another) const;

	__host__ __device__
	float getLengthSq() const;

	__host__ __device__
	float getLength() const;

	__host__ __device__
	Vector2D getUnit() const;

	/// <summary>
	/// ���������� ���θ� �����Ѵ�.
	/// </summary>
	/// <returns>
	/// ���������� ����
	/// </returns>
	__host__ __device__
	bool isZero() const;

	__host__ __device__
	Vector2D operator+(const Vector2D &another) const;

	__host__ __device__
	Vector2D& operator+=(const Vector2D &another);

	__host__ __device__
	Vector2D operator-(const Vector2D &another) const;

	__host__ __device__
	Vector2D operator-() const;

	__host__ __device__
	Vector2D& operator-=(const Vector2D &another);

	__host__ __device__
	Vector2D operator*(float ratio) const;

	__host__ __device__
	Vector2D& operator*=(float ratio);

	__host__ __device__
	Vector2D operator/(float ratio) const;

	__host__ __device__
	Vector2D& operator/=(float ratio);

	/// <summary>
	/// �� ������ �����ϴ� ���� ���� ��� ��ġ�ϴ��� ���θ� �˻��Ѵ�
	/// </summary>
	/// <param name="operand">
	/// �� ���
	/// </param>
	/// <returns>
	/// ��ġ ����
	/// </returns>
	__host__ __device__
	bool operator==(const Vector2D& operand) const;

	/// <summary>
	/// �� ������ �����ϴ� ���� ���� ��ġ���� �ʴ��� ���θ� �˻��Ѵ�
	/// </summary>
	/// <param name="operand">
	/// �� ���
	/// </param>
	/// <returns>
	/// ����ġ ����
	/// </returns>
	__host__ __device__
	bool operator!=(const Vector2D& operand) const;
};

__host__ __device__
Vector2D operator*(float ratio, const Vector2D &vector);
