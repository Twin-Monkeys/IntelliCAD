/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	파일명			: Vector2D.h
*	작성자			: 이세인
*	최종 수정일		: 19.05.30
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
	/// 영벡터인지 여부를 조사한다.
	/// </summary>
	/// <returns>
	/// 영벡터인지 여부
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
	/// 두 벡터의 대응하는 원소 값이 모두 일치하는지 여부를 검사한다
	/// </summary>
	/// <param name="operand">
	/// 비교 대상
	/// </param>
	/// <returns>
	/// 일치 여부
	/// </returns>
	__host__ __device__
	bool operator==(const Vector2D& operand) const;

	/// <summary>
	/// 두 벡터의 대응하는 원소 값이 일치하지 않는지 여부를 검사한다
	/// </summary>
	/// <param name="operand">
	/// 비교 대상
	/// </param>
	/// <returns>
	/// 불일치 여부
	/// </returns>
	__host__ __device__
	bool operator!=(const Vector2D& operand) const;
};

__host__ __device__
Vector2D operator*(float ratio, const Vector2D &vector);
