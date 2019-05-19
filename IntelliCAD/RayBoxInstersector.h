/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	���ϸ�			: RayBoxIntersector.h
*	�ۼ���			: ����
*	���� ������		: 19.03.18
*/

#pragma once

#include "Point3D.h"
#include "Range.hpp"
#include "Size3D.hpp"

namespace RayBoxIntersector 
{
	/* function */
	/// <summary>
	/// <para>�־��� �ȼ����� ī�޶� ���� �������� �þ� ������ �����Ͽ��� ��</para>
	/// <para>������ �����ϴ��� ���θ� �����ϰ�, ���� ������ ����Ѵ�.</para>
	/// </summary>
	/// <param name="volumeSize">
	/// ���� ũ��
	/// </param>
	/// <param name="pixelPosition">
	/// ��ũ�� �ȼ� ��ġ
	/// </param>
	/// <param name="camDirection">
	/// ī�޶� ���� ����
	/// </param>
	/// <returns>
	/// ���� ����
	/// </returns>
	__host__ __device__
	Range<float> getValidRange(
		const Size3D<>& volumeSize,
		const Point3D& pixelPosition, 
		const Vector3D& camDirection);
}