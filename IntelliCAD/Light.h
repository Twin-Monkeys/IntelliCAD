/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	파일명			: Light.h
*	작성자			: 원진
*	최종 수정일		: 19.05.05
*/

#pragma once

#include "Color.hpp"
#include "Point3D.h"

class Light
{
public:
	/* constructor */
	/// <summary>
	/// 생성자
	/// </summary>
	Light() = default;

	/// <summary>
	/// 생성자
	/// </summary>
	/// <param name="ambient">
	/// 주변광 세기
	/// </param>
	/// <param name="diffuse">
	/// 난반사광 세기
	/// </param>
	/// <param name="specular">
	/// 정반사광 세기
	/// </param>
	/// <param name="position">
	/// 조명 위치
	/// </param>
	/// <param name="enabled">
	/// 조명 활성화 유무
	/// </param>
	Light(
		const Color<float>& ambient,
		const Color<float>& diffuse,
		const Color<float>& specular,
		const Point3D& position,
		const bool enabled = false);

	/* member variable */
	/// <summary>
	/// 주변광 세기
	/// </summary>
	Color<float> ambient;

	/// <summary>
	/// 난반사광 세기
	/// </summary>
	Color<float> diffuse;

	/// <summary>
	/// 정반사광 세기
	/// </summary>
	Color<float> specular;

	/// <summary>
	/// 조명 위치
	/// </summary>
	Point3D position;

	/// <summary>
	/// 조명 활성화 여부
	/// </summary>
	bool enabled = false;
};