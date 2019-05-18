/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	파일명			: Light.cpp
*	작성자			: 원진
*	최종 수정일		: 19.05.05
*/

#include "Light.h"

/* constructor */
Light::Light(
	const Color<float>& ambient,
	const Color<float>& diffuse,
	const Color<float>& specular,
	const Point3D& position,
	const bool enabled) :
	ambient(ambient), diffuse(diffuse), specular(specular), position(position), enabled(enabled)
{}