/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	���ϸ�			: OrthoBasis.h
*	�ۼ���			: ����
*	���� ������		: 19.03.07
*/

#pragma once

#include "Vector3D.h"

class OrthoBasis
{
public:
	/* constructor */
	OrthoBasis() = default;

	/// <summary>
	/// ������
	/// </summary>
	/// <param name="u">
	/// ���� ����
	/// </param>
	/// <param name="v">
	/// ���� ���� ����
	/// </param>
	/// <param name="w">
	/// ���� ���� ����
	/// </param>
	/// <returns>
	/// ���� ���� ����
	/// </returns>
	__host__ __device__
	OrthoBasis(
		const Vector3D& u,
		const Vector3D& v,
		const Vector3D& w);

	/* member function */
	/// <summary>
	/// ���� ���� ������ �����Ѵ�.
	/// <param name="u">
	/// ���� ����
	/// </param>
	/// <param name="v">
	/// ���� ���� ����
	/// </param>
	/// <param name="w">
	/// ���� ���� ����
	/// </param>
	/// </summary>
	__host__ __device__
	void set(
		const Vector3D& u,
		const Vector3D& v,
		const Vector3D& w);

	/* member variable */
	/// <summary>
	/// <para>���� ����</para>
	/// <para>(ī�޶� �ٶ󺸴� ��ġ - ī�޶� ��ġ)</para>
	/// </summary>
	Vector3D u;

	/// <summary>
	/// <para>���� ���� ����</para>
	/// <para>���� ���Ϳ� ������ ���簢�� ����� ��ũ���� �ִٰ� �������� ��</para>
	/// <para>��ũ���� ���� �� �������� �����̱� ���� ���� ���ʹ�.</para>
	/// </summary>
	Vector3D v;

	/// <summary>
	/// <para>���� ���� ����</para>
	/// <para>���� ���Ϳ� ������ ���簢�� ����� ��ũ���� �ִٰ� �������� ��</para>
	/// <para>��ũ���� ���� �� �������� �����̱� ���� ���� ���ʹ�.</para>
	/// </summary>
	Vector3D w;
};