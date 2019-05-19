/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	���ϸ�			: Light.cpp
*	�ۼ���			: ����
*	���� ������		: 19.05.05
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