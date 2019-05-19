/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	파일명			: OrthoBasis.cpp
*	작성자			: 원진
*	최종 수정일		: 19.03.07
*/

#include "OrthoBasis.h"

/* constructor */
__host__ __device__
OrthoBasis::OrthoBasis(
	const Vector3D& u,
	const Vector3D& v,
	const Vector3D& w) :
	u(u), v(v), w(w)
{}

/* member function */
__host__ __device__
void OrthoBasis::set(
	const Vector3D& u,
	const Vector3D& v,
	const Vector3D& w) 
{
	this->u = u;
	this->v = v;
	this->w = w;
}