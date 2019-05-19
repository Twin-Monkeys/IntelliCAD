/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	���ϸ�			: Camera.h
*	�ۼ���			: ����
*	���� ������		: 19.03.06
*
*	ī�޶� Ŭ����
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
	/// ������
	/// </summary>
	/// <param name="eye">
	/// ī�޶� ��ġ
	/// </param>
	/// <param name="at">
	/// ī�޶� �ٶ󺸴� ��ġ
	/// </param>
	/// <param name="upVector">
	/// ī�޶� �Ӹ��� ���ϴ� ����
	/// </param>
	Camera(
		const Point3D& eye,
		const Point3D& at,
		const Vector3D& upVector);

	/* member function */
	/// <summary>
	/// ī�޶� ������ �����Ѵ�.
	/// </summary>
	/// <param name="eye">
	/// ī�޶� ��ġ
	/// </param>
	/// <param name="at">
	/// ī�޶� �ٶ󺸴� ��ġ
	/// </param>
	/// <param name="upVector">
	/// ī�޶� �Ӹ��� ���ϴ� ����
	/// </param>
	void set(
		const Point3D& eye,
		const Point3D& at,
		const Vector3D& upVector);

	/// <summary>
	/// ī�޶� ��ġ�� ��ȯ�Ѵ�
	/// </summary>
	/// <returns>
	/// ī�޶� ��ġ
	/// </returns>
	const Point3D& getPosition() const;
	
	/// <summary>
	/// ī�޶��� ���� ������ ��ȯ�Ѵ�
	/// </summary>
	/// <returns>
	/// ���� ����
	/// </returns>
	const OrthoBasis& getOrthoBasis() const;

	/// <summary>
	/// �ٶ󺸴� �������� ���� Ȥ�� �����Ѵ�
	/// </summary>
	/// <param name="delta">
	/// �̵���
	/// </param>
	void adjustForward(const float delta);

	/// <summary>
	/// ���� ���� ����(������ ������)���� �̵��Ѵ�
	/// </summary>
	/// <param name="delta">
	/// �̵���
	/// </param>
	void adjustHorizontal(const float delta);

	/// <summary>
	/// ���� ���� ����(������ ������)���� �̵��Ѵ�
	/// </summary>
	/// <param name="delta">
	/// �̵���
	/// </param>
	void adjustVertical(const float delta);

	/// <summary>
	/// ���� ���� ������ ������ ä�� ��ġ�� �̵��Ѵ�
	/// </summary>
	/// <param name="position">
	/// �̵��� ��ġ
	/// </param>
	void moveTo(const Point3D& position);

	/// <summary>
	/// horiz�� ������ ȸ���Ѵ�
	/// </summary>
	/// <param name="angle">
	/// ȸ���� ���� (����)
	/// </param>
	void adjustPitch(const float angle);

	/// <summary>
	/// vert�� ������ ȸ���Ѵ�
	/// </summary>
	/// <param name="angle">
	/// ȸ���� ���� (����)
	/// </param>
	void adjustYaw(const float angle);

	/// <summary>
	/// dir�� ������ ȸ���Ѵ� 
	/// </summary>
	/// <param name="angle">
	/// ȸ���� ���� (����)
	/// </param>
	void adjustRoll(const float angle);

	/// <summary>
	/// <para>�������� �߽����� �˵� ȸ���� �����Ѵ�</para>
	/// <para>ȸ���� ���� x���� �������� �Ѵ�</para> 
	/// </summary>
	/// <param name="pivot">
	/// ȸ�� ������
	/// </param>
	/// <param name="angle">
	/// ȸ���� ���� (����)
	/// </param>
	void orbitPitch(const Point3D& pivot, const float angle);

	/// <summary>
	/// <para>�������� �߽����� �˵� ȸ���� �����Ѵ�</para>
	/// <para>ȸ���� ���� ���� �������� �Ѵ�</para> 
	/// </summary>
	/// <param name="pivot">
	/// ȸ�� ������
	/// </param>
	/// <param name="angle">
	/// ȸ���� ���� (����)
	/// </param>
	void orbitYaw(const Point3D& pivot, const float angle);

	/// <summary>
	/// <para>�������� �߽����� �˵� ȸ���� �����Ѵ�</para>
	/// <para>ȸ���� ���� z���� �������� �Ѵ�</para> 
	/// </summary>
	/// <param name="pivot">
	/// ȸ�� ������
	/// </param>
	/// <param name="angle">
	/// ȸ���� ���� (����)
	/// </param>
	void orbitRoll(const Point3D& pivot, const float angle);

private:
	/* member variable */
	/// <summary>
	/// ī�޶� ��ġ
	/// </summary>
	Point3D __position;

	/// <summary>
	/// ī�޶��� ���� ����
	/// </summary>
	OrthoBasis __orthoBasis;
};