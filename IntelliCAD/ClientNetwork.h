/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	���ϸ�			: ClientNetwork.h
*	�ۼ���			: ������
*	���� ������		: 19.05.01
*
*	������ ����ϱ� ���� Ŭ���̾�Ʈ ��Ʈ��ũ �ֻ��� ���
*/

#pragma once

#include <WinSock2.h>
#include <memory>
#include <set>
#include "Socket.h"
#include "SystemInitListener.h"
#include "ServerConnectingListener.h"
#include "ConnectionClosedListener.h"
#include "ConnectionCheckListener.h"

/// <summary>
/// <para>������ ����ϱ� ���� Ŭ���̾�Ʈ ��Ʈ��ũ ����̴�.</para>
/// <para>ClientNetwork ��ü�� ���� ������ ���� �ϳ��� ��ü�� ������ �� �ִ�.</para>
/// <para>���� ��ü�� <c>ClientNetwork::getInstance()</c> �Լ��� ���� ��´�.</para>
/// <para>������ �����ϱ� ���ؼ��� ������ ����ϱ� ���� <c>Socket</c> ��ü�� ���� �����ؾ� �Ѵ�.</para>
/// <para><c>Socket</c> ��ü�� �����Ϸ��� <c>ClientNetwork::createClientSocket()</c> �Լ��� �̿��Ѵ�.</para>
/// <para><c>Socket</c> ��ü�� ���� �Ǿ��ٸ� <c>ClientNetwork::connect()</c> �Լ��� ���� ������ �����Ѵ�.</para>
/// </summary>
class ClientNetwork :
	public SystemInitListener, public ServerConnectingListener,
	public ConnectionCheckListener, public ConnectionClosedListener
{
private:
	WSADATA __wsaData;

	bool __isRun = false;

	/// <summary>
	/// ���� ���� �����ϴ� ���� ��ü�̴�.
	/// </summary>
	static ClientNetwork __instance;

	std::tstring __serverIP = _T("0.0.0.0");
	std::tstring __serverPort = _T("9000");

	/// <summary>
	/// �׻� ������ ����Ǿ��ִ� ����
	/// </summary>
	std::shared_ptr<Socket> __pSocket = nullptr;


	// ���� ó�� ���� �������, �޼���
	
	std::set<std::shared_ptr<Socket>> __tempSockets;

	std::set<std::shared_ptr<Socket>> __taskCompletedSockets;

	std::shared_ptr<Socket> getSocket();

	//

	/// <summary>
	/// <para>���� ��ü�� �����ϱ� ���� ���ο� �⺻ ������.</para>
	/// <para>��Ʈ��ũ ��Ű� ������ �ʱ�ȭ ������ ���Եȴ�.</para>
	/// </summary>
	ClientNetwork();

	ClientNetwork(const ClientNetwork &) = delete;
	ClientNetwork(ClientNetwork &&) = delete;

public:
	/// <summary>
	/// <para>���������� ����� ������ IP ���� ��ȯ�Ѵ�.</para>
	/// <para>�ѹ��� ����� ���� ���� ��� 0.0.0.0�� ��ȯ�Ѵ�.</para>
	/// </summary>
	/// <returns>���������� ����� ������ IP ��</returns>
	const std::tstring& getServerIP() const;

	/// <summary>
	/// <para>���������� ����� ������ ��Ʈ ���� ��ȯ�Ѵ�.</para>
	/// <para>�ѹ��� ����� ���� ���� ��� 9000�� ��ȯ�Ѵ�.</para>
	/// </summary>
	/// <returns>���������� ����� ������ ��Ʈ ��</returns>
	const std::tstring& getServerPort() const;

	/// <summary>
	/// <para><c>ClientSocket</c> ��ü�� �����Ǿ� �ִ��� ���θ� ��ȯ�Ѵ�.</para>
	/// <para><c>ClientSocket</c> ��ü�� <c>ClientNetwork::createClientSocket()</c> �Լ��� ���� ���� �� �ִ�.</para>
	/// </summary>
	/// <returns><c>ClientSocket</c> ��ü�� �����Ǿ� �ִ��� ����</returns>
	bool isCreated() const;

	/// <summary>
	/// <para>������ ������ �Ǿ��ִ� �������� ���θ� ��ȯ�Ѵ�.</para>
	/// <para>�������� ������ <c>Socket</c> ��ü�� ������ �� <c>ClientNetwork::connect()</c> �Լ��� �̿��Ѵ�.</para>
	/// </summary>
	/// <returns>���� ���� ����</returns>
	bool isConnected() const;

	/// <summary>
	/// <para>������ ����ϱ� ���� <c>Socket</c> ��ü�� �����Ѵ�.</para>
	/// <para>���ڷδ� ������ IP�� ��Ʈ�� �ʿ��ϴ�.</para>
	/// <para>��ü ���� ������ true��, �������� ������ ���� ���� �� false�� ��ȯ�Ѵ�.</para>
	/// <para>���� ������ ����ϱ� ���ؼ��� <c>ClientNetwork::connect()</c> �Լ��� ���� ������ ���� �۾��� �ʿ��ϴ�.</para>
	/// </summary>
	/// <param name="serverIP">������ IP</param>
	/// <param name="serverPort">������ ��Ʈ</param>
	/// <returns><c>Socket</c> ��ü ���� ���</returns>
	bool createClientSocket(const std::tstring &serverIP, const std::tstring &serverPort);

	virtual void onServerConnectionResult(std::shared_ptr<Socket> pSocket) override;
	virtual void onSystemInit() override;

	// ���� �߰�
	virtual void onConnectionCheck() override;

	virtual void onConnectionClosed(std::shared_ptr<Socket> pSocket) override;

	/// <summary>
	/// <para><c>ClientNetwork::createClientSocket()</c> �Լ��� �̿��Ͽ� <c>Socket</c> ��ü�� ������ ��, ������ ������ �õ��Ѵ�.</para>
	/// <para>���� ���� �� true��, ���ڷ� �־��� ���� �ð� ���� ������ ������ ������ �� ������ false�� ��ȯ�Ѵ�.</para>
	/// <para>���� �ð��� �и������� �����̴�.</para>
	/// </summary>
	/// <param name="timeout">���� ���� �õ� �� �ִ� ��� �ð�</param>
	/// <returns>������ ���� ���� ����</returns>
	bool connect(int timeout = 5000);

	/// <summary>
	/// <para>�������� ������ �����Ѵ�.</para>
	/// <para>���������� ���ᰡ �� �� true��, ����Ǿ� ���� ���� �����̰ų� ������������ ����Ǹ� false�� ��ȯ�Ѵ�.</para>
	/// </summary>
	/// <returns>������ ���� ���� ����</returns>
	bool close();

	/// <summary>
	/// ClientNetwork�� ���� ��ü�� ��ȯ�Ѵ�.
	/// </summary>
	/// <returns>ClientNetwork ��ü</returns>
	static ClientNetwork& getInstance();

	/// <summary>
	/// <para>�Ҹ����̴�.</para>
	/// <para>��Ʈ��ũ ��Ű� ������ ������ ������ ���Եȴ�.</para>
	/// </summary>
	~ClientNetwork();

	// �񵿱� send
	void sendMSG(const char* const msg, const ProtocolType protocolType);
	void sendObj(Serializable & obj, const ObjectType objectType);
	void sendFile(const std::string &path);
};