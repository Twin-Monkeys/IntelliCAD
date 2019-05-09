/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	���ϸ�			: NetworkUtility.h
*	�ۼ���			: �̼���
*	���� ������		: 19.03.22
*
*	��Ʈ��ũ�� ��ƿ��Ƽ �Լ� ����
*/

#pragma once

#include "tstring.h"

/// <summary>
/// ��Ʈ��ũ�� ������ ��ƿ �Լ��� ��Ƴ��� ���ӽ����̽�
/// </summary>
namespace NetworkUtility
{
	/// <summary>
	/// <para>�־��� ���ڿ��� IP ������ �����ϴ��� �˻��Ѵ�.</para>
	/// <para>IP ������ IPv4 �԰��� ������.</para>
	/// </summary>
	/// <param name="ipString">�˻� ���</param>
	/// <returns>�־��� ���ڿ��� IP ������ �����ϴ��� ����</returns>
	bool checkIPValidation(const std::tstring &ipString);

	/// <summary>
	/// �־��� ���ڿ��� ��Ʈ ������ �����ϴ��� �˻��Ѵ�.
	/// </summary>
	/// <param name="ipString">�˻� ���</param>
	/// <returns>�־��� ���ڿ��� ��Ʈ ������ �����ϴ��� ����</returns>
	bool checkPortValidation(const std::tstring &portString);
};
