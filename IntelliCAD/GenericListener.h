/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	���ϸ�			: GenericListener.h
*	�ۼ���			: �̼���
*	���� ������		: 19.04.06
*
*	���� ���� �̺�Ʈ ������
*/

#pragma once

/// <summary>
/// �������� ����ϱ� ���� generic �̺�Ʈ�� ���� ������ �������̽��̴�.
/// </summary>
class GenericListener
{
public:
	/// <summary>
	/// generic �̺�Ʈ�� ó���ϴ� �Լ��̴�.
	/// </summary>
	virtual void onGeneric() = 0;
};