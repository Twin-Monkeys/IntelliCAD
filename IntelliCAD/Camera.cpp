/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	파일명			: Camera.cpp
*	작성자			: 원진
*	최종 수정일		: 19.03.06
*
*	카메라 클래스
*/

#include "Camera.h"

/* constructor */

Camera::Camera(const float &imgBasedSamplingStep) :
	__imgBasedSamplingStep(imgBasedSamplingStep)
{}

Camera::Camera(
	const float &imageBasedSamplingStep,
	const Point3D& eye,
	const Point3D& at,
	const Vector3D& upVector) :
	Camera(imageBasedSamplingStep)
{
	set(eye, at, upVector);
}

/* member function */
void Camera::set(
	const Point3D& eye,
	const Point3D& at,
	const Vector3D& upVector) 
{
	__position = eye;
	__orthoBasis.u = (at - eye).getUnit();
	__orthoBasis.v = upVector.cross(__orthoBasis.u).getUnit();
	__orthoBasis.w = __orthoBasis.v.cross(__orthoBasis.u).getUnit();
}

const Point3D& Camera::getPosition() const
{
	return __position;
}

const OrthoBasis& Camera::getOrthoBasis() const
{
	return __orthoBasis;
}

void Camera::adjustForward(const float delta)
{
	__position += (__orthoBasis.u * delta);
}

void Camera::adjustHorizontal(const float delta)
{
	__position += (__orthoBasis.v * (delta * __imgBasedSamplingStep));
}

void Camera::adjustVertical(const float delta)
{
	__position += (__orthoBasis.w * (delta * __imgBasedSamplingStep));
}

void Camera::moveTo(const Point3D& position)
{
	__position = position;
}

void Camera::adjustPitch(const float angle)
{
	const float ADJ_ANGLE = (angle * __imgBasedSamplingStep);

	__orthoBasis.u = __orthoBasis.u.rotate(__orthoBasis.v, ADJ_ANGLE).getUnit();
	__orthoBasis.w = __orthoBasis.v.cross(__orthoBasis.u).getUnit();
}

void Camera::adjustYaw(const float angle)
{
	const float ADJ_ANGLE = (angle * __imgBasedSamplingStep);

	__orthoBasis.u = __orthoBasis.u.rotate(__orthoBasis.w, ADJ_ANGLE).getUnit();
	__orthoBasis.v = __orthoBasis.v.rotate(__orthoBasis.w, ADJ_ANGLE).getUnit();
	__orthoBasis.w = __orthoBasis.v.cross(__orthoBasis.u).getUnit();
}

void Camera::adjustRoll(const float angle)
{
	const float ADJ_ANGLE = (angle * __imgBasedSamplingStep);

	__orthoBasis.v = __orthoBasis.v.rotate(__orthoBasis.u, ADJ_ANGLE).getUnit();
	__orthoBasis.w = __orthoBasis.v.cross(__orthoBasis.u).getUnit();
}

void Camera::orbitPitch(const Point3D& pivot, const float angle)
{
	const float ADJ_ANGLE = (angle * __imgBasedSamplingStep);

	__position -= pivot;
	__position = __position.rotate(__orthoBasis.v, ADJ_ANGLE);
	__position += pivot;

	__orthoBasis.u = __orthoBasis.u.rotate(__orthoBasis.v, ADJ_ANGLE).getUnit();
	__orthoBasis.w = __orthoBasis.v.cross(__orthoBasis.u).getUnit();
}

void Camera::orbitYaw(const Point3D& pivot, const float angle)
{
	const float ADJ_ANGLE = (angle * __imgBasedSamplingStep);

	__position -= pivot;
	__position = __position.rotate(__orthoBasis.w, ADJ_ANGLE);
	__position += pivot;

	__orthoBasis.u = __orthoBasis.u.rotate(__orthoBasis.w, ADJ_ANGLE).getUnit();
	__orthoBasis.v = __orthoBasis.v.rotate(__orthoBasis.w, ADJ_ANGLE).getUnit();
	__orthoBasis.w = __orthoBasis.v.cross(__orthoBasis.u).getUnit();
}

void Camera::orbitRoll(const Point3D& pivot, const float angle)
{
	const float ADJ_ANGLE = (angle * __imgBasedSamplingStep);

	__position -= pivot;
	__position = __position.rotate(__orthoBasis.u, ADJ_ANGLE);
	__position += pivot;

	__orthoBasis.v = __orthoBasis.v.rotate(__orthoBasis.u, ADJ_ANGLE).getUnit();
	__orthoBasis.w = __orthoBasis.v.cross(__orthoBasis.u).getUnit();
}