/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	파일명			: OrthoBasis.h
*	작성자			: 원진
*	최종 수정일		: 19.03.07
*/

#pragma once

#include "Vector3D.h"

class OrthoBasis
{
public:
	/* constructor */
	OrthoBasis() = default;

	/// <summary>
	/// 생성자
	/// </summary>
	/// <param name="u">
	/// 시점 벡터
	/// </param>
	/// <param name="v">
	/// 수평 방향 벡터
	/// </param>
	/// <param name="w">
	/// 수직 방향 벡터
	/// </param>
	/// <returns>
	/// 정규 직교 기저
	/// </returns>
	__host__ __device__
	OrthoBasis(
		const Vector3D& u,
		const Vector3D& v,
		const Vector3D& w);

	/* member function */
	/// <summary>
	/// 정규 직교 기저를 설정한다.
	/// <param name="u">
	/// 시점 벡터
	/// </param>
	/// <param name="v">
	/// 수평 방향 벡터
	/// </param>
	/// <param name="w">
	/// 수직 방향 벡터
	/// </param>
	/// </summary>
	__host__ __device__
	void set(
		const Vector3D& u,
		const Vector3D& v,
		const Vector3D& w);

	/* member variable */
	/// <summary>
	/// <para>시점 벡터</para>
	/// <para>(카메라가 바라보는 위치 - 카메라 위치)</para>
	/// </summary>
	Vector3D u;

	/// <summary>
	/// <para>수평 방향 벡터</para>
	/// <para>시점 벡터와 수직인 직사각형 모양의 스크린이 있다고 가정했을 때</para>
	/// <para>스크린의 가로 축 방향으로 움직이기 위한 단위 벡터다.</para>
	/// </summary>
	Vector3D v;

	/// <summary>
	/// <para>수직 방향 벡터</para>
	/// <para>시점 벡터와 수직인 직사각형 모양의 스크린이 있다고 가정했을 때</para>
	/// <para>스크린의 세로 축 방향으로 움직이기 위한 단위 벡터다.</para>
	/// </summary>
	Vector3D w;
};