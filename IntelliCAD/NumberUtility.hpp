/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	���ϸ�			: NumberUtility.hpp
*	�ۼ���			: �̼���
*	���� ������		: 19.03.22
*
*	���� ���õ� ������ �����ϴ� ��ƿ��Ƽ
*/

#pragma once

/// <summary>
/// ���� ������ ��ƿ �Լ��� ��Ƴ��� ���ӽ����̽�
/// </summary>
namespace NumberUtility
{
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
	T truncate(T value, T lower, T upper);

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
	bool isInOfBound(const T value, const T lowerInc, const T upperExc)
	{
		return !isOutOfBound(value, lowerInc, upperExc);
	}

	template <typename T>
	T truncate(const T value, const T lower, const T upper)
	{
		if (value < lower)
			return lower;
		else if (value >= upper)
			return upper;

		return value;
	}
};
