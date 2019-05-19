/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	���ϸ�			: Light.h
*	�ۼ���			: ����
*	���� ������		: 19.05.05
*/

#pragma once

#include "Color.hpp"
#include "Point3D.h"

class Light
{
public:
	/* constructor */
	/// <summary>
	/// ������
	/// </summary>
	Light() = default;

	/// <summary>
	/// ������
	/// </summary>
	/// <param name="ambient">
	/// �ֺ��� ����
	/// </param>
	/// <param name="diffuse">
	/// ���ݻ籤 ����
	/// </param>
	/// <param name="specular">
	/// ���ݻ籤 ����
	/// </param>
	/// <param name="position">
	/// ���� ��ġ
	/// </param>
	/// <param name="enabled">
	/// ���� Ȱ��ȭ ����
	/// </param>
	Light(
		const Color<float>& ambient,
		const Color<float>& diffuse,
		const Color<float>& specular,
		const Point3D& position,
		const bool enabled = false);

	/* member variable */
	/// <summary>
	/// �ֺ��� ����
	/// </summary>
	Color<float> ambient;

	/// <summary>
	/// ���ݻ籤 ����
	/// </summary>
	Color<float> diffuse;

	/// <summary>
	/// ���ݻ籤 ����
	/// </summary>
	Color<float> specular;

	/// <summary>
	/// ���� ��ġ
	/// </summary>
	Point3D position;

	/// <summary>
	/// ���� Ȱ��ȭ ����
	/// </summary>
	bool enabled = false;
};