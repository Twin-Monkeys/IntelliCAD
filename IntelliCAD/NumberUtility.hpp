/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: NumberUtility.hpp
*	작성자			: 원진, 이세인
*	최종 수정일		: 19.03.22
*/

#pragma once

#include "Vector3D.h"
#include "Range.hpp"

#define SQUARE(X) ((X) * (X))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

/// <summary>
/// 수와 관련한 유틸 함수를 모아놓은 네임스페이스
/// </summary>
namespace NumberUtility
{
	/* function */
	/// <summary>
	/// <para>주어진 값 value가 [lowerInc, upperExc) 범위를 벗어나는 지 검사한다.</para>
	/// <para>범위 경계의 하한 값인 lowerInc은 범위에 포함된다.</para>
	/// <para>범위 경계의 상한 값인 upperExc은 범위에 포함되지 않는다.</para>
	/// <para>범위를 벗어나는 경우 true, 벗어나지 않으면 false를 반환한다.</para>
	/// 
	/// </summary>
	/// <param name="value">범위를 벗어나는지 조사할 값</param>
	/// <param name="lowerInc">범위 경계의 하한 값</param>
	/// <param name="upperExc">범위 경계의 상한 값</param>
	/// <returns>주어진 값이 범위를 벗어나는지 여부</returns>
	template <typename T>
	__host__ __device__
	bool isOutOfBound(const T value, const T lowerInc, const T upperExc);

	/// <summary>
	/// <para>주어진 값 value가 [lowerInc, upperExc) 범위에 포함되는지 검사한다.</para>
	/// <para>범위 경계의 하한 값인 lowerInc은 범위에 포함된다.</para>
	/// <para>범위 경계의 상한 값인 upperExc은 범위에 포함되지 않는다.</para>
	/// <para>범위에 포함되는 경우 true, 포함되지 않으면 false를 반환한다.</para>
	/// 
	/// </summary>
	/// <param name="value">범위에 포함되는지 조사할 값</param>
	/// <param name="lowerInc">범위 경계의 하한 값</param>
	/// <param name="upperExc">범위 경계의 상한 값</param>
	/// <returns>주어진 값이 범위에 포함되는지 여부</returns>
	template <typename T>
	__host__ __device__
	bool isInOfBound(const T value, const T lowerInc, const T upperExc);

	/// <summary>
	/// <para>주어진 값 value가 [lower, upper] 범위에 존재하도록 강제 조정한다.</para>
	/// <para>value가 범위에 포함되는 경우 value를 그대로 반환한다.</para>
	/// <para>value가 lower보다 작은 값을 가지는 경우 lower를 반환한다.</para>
	/// <para>value가 upper보다 큰 값을 가지는 경우 upper를 반환한다.</para>
	/// </summary>
	/// <param name="value">범위 내에 존재하도록 조정할 값</param>
	/// <param name="lower">범위 경계의 하한 값</param>
	/// <param name="upper">범위 경계의 상한 값</param>
	/// <returns>범위 내로 조정된 값</returns>
	template <typename T>
	__host__ __device__
	T truncate(const T value, const T lower, const T upper);

	/// <summary>
	/// 두 값을 유사 동등 비교한다
	/// </summary>
	/// <param name="operand1">
	/// 비교 대상1
	/// </param>
	/// <param name="operand2">
	/// 비교 대상2
	/// </param>
	/// <param name="epsilon">
	/// 오차 한계
	/// </param>
	/// <returns>
	/// 유사 동등 여부
	/// </returns>
	__host__ __device__
	bool nearEqual(
		const float operand1, const float operand2, const float epsilon = 1e-6f);

	/// <summary>
	/// <para>2개의 3차원 벡터 정보를 입력 받아 역방향의 그라디언트를 계산한다.</para>
	/// <para>역방향은 정방향 그라디언트의 반대 부호를 가진다.</para>
	/// </summary>
	/// <param name="operand1X">
	/// 좌측 벡터의 x 값
	/// </param>
	/// <param name="operand1Y">
	/// 좌측 벡터의 y 값 
	/// </param>
	/// <param name="operand1Z">
	/// 좌측 벡터의 z 값
	/// </param>
	/// <param name="operand2X">
	/// 우측 벡터의 x 값
	/// </param>
	/// <param name="operand2Y">
	/// 우측 벡터의 y 값
	/// </param>
	/// <param name="operand2Z">
	/// 우측 벡터의 z 값
	/// </param>
	/// <returns>
	/// 역방향 그라디언트
	/// </returns>
	__host__ __device__
	Vector3D inverseGradient(
		const float operand1X, const float operand2X,
		const float operand1Y, const float operand2Y,
		const float operand1Z, const float operand2Z);

	template <typename T>
	bool isOutOfBound(const T value, const T lowerInc, const T upperExc)
	{
		if (value < lowerInc)
			return true;
		else if (value >= upperExc)
			return true;

		return false;
	}

	template <typename T>
	__host__ __device__
	bool isInOfBound(const T value, const T lowerInc, const T upperExc)
	{
		return !isOutOfBound(value, lowerInc, upperExc);
	}

	template <typename T>
	__host__ __device__
	T truncate(const T value, const T lower, const T upper)
	{
		if (value <= lower)
			return lower;
		else if (value >= upper)
			return upper;

		return value;
	}
};
