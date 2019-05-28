/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	���ϸ�			: NumberUtility.hpp
*	�ۼ���			: ����, �̼���
*	���� ������		: 19.03.22
*/

#pragma once

#include "Vector3D.h"
#include "Range.hpp"

#define SQUARE(X) ((X) * (X))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

/// <summary>
/// ���� ������ ��ƿ �Լ��� ��Ƴ��� ���ӽ����̽�
/// </summary>
namespace NumberUtility
{
	/* function */
	/// <summary>
	/// <para>�־��� �� value�� [lowerInc, upperExc) ������ ����� �� �˻��Ѵ�.</para>
	/// <para>���� ����� ���� ���� lowerInc�� ������ ���Եȴ�.</para>
	/// <para>���� ����� ���� ���� upperExc�� ������ ���Ե��� �ʴ´�.</para>
	/// <para>������ ����� ��� true, ����� ������ false�� ��ȯ�Ѵ�.</para>
	/// 
	/// </summary>
	/// <param name="value">������ ������� ������ ��</param>
	/// <param name="lowerInc">���� ����� ���� ��</param>
	/// <param name="upperExc">���� ����� ���� ��</param>
	/// <returns>�־��� ���� ������ ������� ����</returns>
	template <typename T>
	__host__ __device__
	bool isOutOfBound(T value, T lowerInc, T upperExc);

	/// <summary>
	/// <para>�־��� �� value�� [lowerInc, upperExc) ������ ���ԵǴ��� �˻��Ѵ�.</para>
	/// <para>���� ����� ���� ���� lowerInc�� ������ ���Եȴ�.</para>
	/// <para>���� ����� ���� ���� upperExc�� ������ ���Ե��� �ʴ´�.</para>
	/// <para>������ ���ԵǴ� ��� true, ���Ե��� ������ false�� ��ȯ�Ѵ�.</para>
	/// 
	/// </summary>
	/// <param name="value">������ ���ԵǴ��� ������ ��</param>
	/// <param name="lowerInc">���� ����� ���� ��</param>
	/// <param name="upperExc">���� ����� ���� ��</param>
	/// <returns>�־��� ���� ������ ���ԵǴ��� ����</returns>
	template <typename T>
	__host__ __device__
	bool isInOfBound(T value, T lowerInc, T upperExc);

	/// <summary>
	/// <para>�־��� �� value�� [lower, upper] ������ �����ϵ��� ���� �����Ѵ�.</para>
	/// <para>value�� ������ ���ԵǴ� ��� value�� �״�� ��ȯ�Ѵ�.</para>
	/// <para>value�� lower���� ���� ���� ������ ��� lower�� ��ȯ�Ѵ�.</para>
	/// <para>value�� upper���� ū ���� ������ ��� upper�� ��ȯ�Ѵ�.</para>
	/// </summary>
	/// <param name="value">���� ���� �����ϵ��� ������ ��</param>
	/// <param name="lower">���� ����� ���� ��</param>
	/// <param name="upper">���� ����� ���� ��</param>
	/// <returns>���� ���� ������ ��</returns>
	template <typename T>
	__host__ __device__
	T truncate(T value, T lower, T upper);

	/// <summary>
	/// �� ���� ���� ���� ���Ѵ�
	/// </summary>
	/// <param name="operand1">
	/// �� ���1
	/// </param>
	/// <param name="operand2">
	/// �� ���2
	/// </param>
	/// <param name="epsilon">
	/// ���� �Ѱ�
	/// </param>
	/// <returns>
	/// ���� ���� ����
	/// </returns>
	__host__ __device__
	bool nearEqual(
		const float operand1, const float operand2, const float epsilon = 1e-5f);

	/// <summary>
	/// <para>2���� 3���� ���� ������ �Է� �޾� �������� �׶���Ʈ�� ����Ѵ�.</para>
	/// <para>�������� ������ �׶���Ʈ�� �ݴ� ��ȣ�� ������.</para>
	/// </summary>
	/// <param name="operand1X">
	/// ���� ������ x ��
	/// </param>
	/// <param name="operand1Y">
	/// ���� ������ y �� 
	/// </param>
	/// <param name="operand1Z">
	/// ���� ������ z ��
	/// </param>
	/// <param name="operand2X">
	/// ���� ������ x ��
	/// </param>
	/// <param name="operand2Y">
	/// ���� ������ y ��
	/// </param>
	/// <param name="operand2Z">
	/// ���� ������ z ��
	/// </param>
	/// <returns>
	/// ������ �׶���Ʈ
	/// </returns>
	__host__ __device__
	Vector3D inverseGradient(
		const float operand1X, const float operand2X,
		const float operand1Y, const float operand2Y,
		const float operand1Z, const float operand2Z);

	template <typename T>
	bool isOutOfBound(const T value, const T lowerInc, const T upperExc)
	{
		if (value < lowerInc)
			return true;
		else if (value >= upperExc)
			return true;

		return false;
	}

	template <typename T>
	__host__ __device__
	bool isInOfBound(const T value, const T lowerInc, const T upperExc)
	{
		return !isOutOfBound(value, lowerInc, upperExc);
	}

	template <typename T>
	__host__ __device__
	T truncate(const T value, const T lower, const T upper)
	{
		if (value < lower)
			return lower;
		else if (value >= upper)
			return upper;

		return value;
	}
};
