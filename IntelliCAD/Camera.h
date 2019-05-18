/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	파일명			: Camera.h
*	작성자			: 원진
*	최종 수정일		: 19.03.06
*
*	카메라 클래스
*/

#pragma once

#include "Point3D.h"
#include "OrthoBasis.h"

class Camera 
{
public:
	/* constructor */
	Camera() = default;

	/// <summary>
	/// 생성자
	/// </summary>
	/// <param name="eye">
	/// 카메라 위치
	/// </param>
	/// <param name="at">
	/// 카메라가 바라보는 위치
	/// </param>
	/// <param name="upVector">
	/// 카메라 머리가 향하는 방향
	/// </param>
	Camera(
		const Point3D& eye,
		const Point3D& at,
		const Vector3D& upVector);

	/* member function */
	/// <summary>
	/// 카메라 정보를 설정한다.
	/// </summary>
	/// <param name="eye">
	/// 카메라 위치
	/// </param>
	/// <param name="at">
	/// 카메라가 바라보는 위치
	/// </param>
	/// <param name="upVector">
	/// 카메라 머리가 향하는 방향
	/// </param>
	void set(
		const Point3D& eye,
		const Point3D& at,
		const Vector3D& upVector);

	/// <summary>
	/// 카메라 위치를 반환한다
	/// </summary>
	/// <returns>
	/// 카메라 위치
	/// </returns>
	const Point3D& getPosition() const;
	
	/// <summary>
	/// 카메라의 직교 기저를 반환한다
	/// </summary>
	/// <returns>
	/// 직교 기저
	/// </returns>
	const OrthoBasis& getOrthoBasis() const;

	/// <summary>
	/// 바라보는 방향으로 전진 혹은 후진한다
	/// </summary>
	/// <param name="delta">
	/// 이동량
	/// </param>
	void adjustForward(const float delta);

	/// <summary>
	/// 눈의 수평 방향(기저의 수평축)으로 이동한다
	/// </summary>
	/// <param name="delta">
	/// 이동량
	/// </param>
	void adjustHorizontal(const float delta);

	/// <summary>
	/// 눈의 수직 방향(기저의 수직축)으로 이동한다
	/// </summary>
	/// <param name="delta">
	/// 이동량
	/// </param>
	void adjustVertical(const float delta);

	/// <summary>
	/// 눈이 직교 기저를 유지한 채로 위치만 이동한다
	/// </summary>
	/// <param name="position">
	/// 이동할 위치
	/// </param>
	void moveTo(const Point3D& position);

	/// <summary>
	/// horiz를 축으로 회전한다
	/// </summary>
	/// <param name="angle">
	/// 회전할 각도 (라디안)
	/// </param>
	void adjustPitch(const float angle);

	/// <summary>
	/// vert를 축으로 회전한다
	/// </summary>
	/// <param name="angle">
	/// 회전할 각도 (라디안)
	/// </param>
	void adjustYaw(const float angle);

	/// <summary>
	/// dir를 축으로 회전한다 
	/// </summary>
	/// <param name="angle">
	/// 회전할 각도 (라디안)
	/// </param>
	void adjustRoll(const float angle);

	/// <summary>
	/// <para>기준점을 중심으로 궤도 회전을 수행한다</para>
	/// <para>회전은 로컬 x축을 기준으로 한다</para> 
	/// </summary>
	/// <param name="pivot">
	/// 회전 기준점
	/// </param>
	/// <param name="angle">
	/// 회전할 각도 (라디안)
	/// </param>
	void orbitPitch(const Point3D& pivot, const float angle);

	/// <summary>
	/// <para>기준점을 중심으로 궤도 회전을 수행한다</para>
	/// <para>회전은 로컬 축을 기준으로 한다</para> 
	/// </summary>
	/// <param name="pivot">
	/// 회전 기준점
	/// </param>
	/// <param name="angle">
	/// 회전할 각도 (라디안)
	/// </param>
	void orbitYaw(const Point3D& pivot, const float angle);

	/// <summary>
	/// <para>기준점을 중심으로 궤도 회전을 수행한다</para>
	/// <para>회전은 로컬 z축을 기준으로 한다</para> 
	/// </summary>
	/// <param name="pivot">
	/// 회전 기준점
	/// </param>
	/// <param name="angle">
	/// 회전할 각도 (라디안)
	/// </param>
	void orbitRoll(const Point3D& pivot, const float angle);

private:
	/* member variable */
	/// <summary>
	/// 카메라 위치
	/// </summary>
	Point3D __position;

	/// <summary>
	/// 카메라의 직교 기저
	/// </summary>
	OrthoBasis __orthoBasis;
};