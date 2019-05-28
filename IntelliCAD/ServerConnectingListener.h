/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	���ϸ�			: ServerConnectingListener.h
*	�ۼ���			: �̼���
*	���� ������		: 19.03.06
*
*	������ ������ �Ǹ� ���� ����� ��ȯ�޴� ������
*/

#pragma once

#include "Socket.h"

/// <summary>
/// <para>ServerConnected �̺�Ʈ�� ���� ó�� �ɷ��� �䱸�ϴ� �������̽��̴�.</para>
/// <para>�� �������̽��� ��� �� ������ �� EventBroadcaster�� ����ϸ� �̺�Ʈ�� ���� �ݹ��� �ڵ����� �̷������.</para>
/// <para>�̺�Ʈ ó���� �׻� ���� �����带 ���� �̷������ �Ѵ�.</para>
/// <para>��, ���� ���ϰ� ū �۾��� �ʿ��ϴٸ� �ݵ�� AsyncTaskManager�� ���� �񵿱� ó�����־�� �Ѵ�.</para>
/// </summary>
class ServerConnectingListener
{
public:
	/// <summary>
	/// <para>�������� ���� �õ� ���� �� ����� �ݹ�޴� �Լ��̴�.</para>
	/// <para>�� �Լ��� ���� ���� �Լ��̹Ƿ� ServerConnectingListener �������̽��� �����ϴ� Ŭ������</para>
	/// <para>�� �Լ��� ���� ó�� ��ƾ�� �ݵ�� �����Ͽ��� �Ѵ�.</para>
	/// <para>�Ķ���ͷ� ���޵Ǵ� ��Ŷ�� ������ ����� ��Ŷ ��ü�̴�.</para>
	/// <para>���� ������ �߸� �Ǿ��ų� ������ �������� ���� ��� nullptr�� ���޵ȴ�.</para>
	/// </summary>
	/// <param name="pSocket">������ ����� �� ȹ���� ���� ��ü</param>
	virtual void onServerConnectionResult(std::shared_ptr<Socket> pSocket) = 0;
};