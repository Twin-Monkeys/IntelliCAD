/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	���ϸ�			: Socket.h
*	�ۼ���			: ������
*	���� ������		: 19.05.01
*
*	�ܴ�� ����� ���� ���� ���
*/

#pragma once

#include <WinSock2.h>
#include <memory>
#include "tstring.h"
#include "PacketHeader.h"
#include "Serializable.h"


/// <summary>
/// <para>�ܴ�� ����� ���� ���� Ŭ�����̴�.</para>
/// <para>������ Ŭ���̾�Ʈ ���ʿ��� ���������� ���̴� Ŭ�����̴�.</para>
/// <para>��ü�� ������ �����ڸ� ���� ȣ���ϰų�, <c>Socket::create()</c> �Լ��� �̿��Ѵ�.</para>
/// <para><c>Socket::create()</c> �Լ��� Ŭ���̾�Ʈ �ܿ����� ���̸�, ���� <c>Socket::connect()</c>�� ���� ������ �����Ͽ��� �Ѵ�.</para>
/// </summary>
class Socket : public std::enable_shared_from_this<Socket>
{
private:

	std::tstring __filePath;

	bool __isTempSocket = false;

	bool __isReceving = false;
	bool __isSending = false;

	friend class NetworkStream;
	friend class ClientNetwork;

	SOCKET __sock;

	SOCKADDR_IN __sockAddr;

	bool __connected;

	Socket(const Socket &) = delete;
	Socket(Socket &&) = delete;


	//���� recv & send part

	static const int BUF_SIZE = 1024;

	int send(const char* const data, const size_t size);
	int recv(char* const p, const size_t size);

	char bufForPH[12];
	char bufForTemp[BUF_SIZE];

	std::tstring recvMSG();

	/// <summary>
	/// Serializable �� ����� Object�� ��Ʈ��ũ�κ��� recv�Ѵ�.
	/// recv�� �����͸� ������ Object�� �����Ѵ�.
	/// ������ Object�� notify�Ͽ� Object�� �����ϴ� ���� Listener ž�� Object���� �����Ѵ�.
	/// notify �� ���޵Ǵ� ���� Object�� shared_ptr �̴�.
	/// </summary>
	bool recvObj(const ObjectType objectType, const uint32_t byteCount); // notify �ʿ�

	bool recvFile(const std::tstring t_path, const uint32_t byteCount); // notify �ʿ�

	const PacketHeader* getHeader();

	bool sendHeader(PacketHeader ph);

public:
	void setFilePath(std::tstring & filePath);

	bool isReceving() const;
	bool isSending() const;

	/// <summary>
	/// ���� Ŭ������ ����� �������̴�.
	/// </summary>
	/// <param name="hSockRaw">raw data</param>
	/// <param name="sockAddr">raw data</param>
	/// <param name="connected">��� �ܸ����� ���� ����</param>
	Socket(const SOCKET &hSockRaw, const SOCKADDR_IN &sockAddr, bool connected, bool isTempSocket);

	/// <summary>
	/// ���� ��ü�� ��� �ܸ��� ����Ǿ� �ִ��� ���θ� ��ȯ�Ѵ�.
	/// </summary>
	/// <returns>��� �ܸ��� ����Ǿ� �ִ��� ����</returns>
	bool isConnected() const;

	/// <summary>
	/// <para>��� �ܸ��� ������ �õ��Ѵ�.</para>
	/// <para>���ڷ� ���ῡ ���� ���� �ð��� �Է��Ѵ�.</para>
	/// <para>���ῡ ���� �� ���� ��ü�� ���� �����͸�, ���ῡ �����ϰų� ���� �ð��� ���� �� nullptr�� ��ȯ�Ѵ�.</para>
	/// </summary>
	/// <param name="timeout">���� �õ��� ���� ���ѽð�</param>
	/// <returns>���ῡ ���� �� ���� ��ü�� ���� ������, ���ῡ �����ϰų� ���� �ð��� ���� �� nullptr</returns>
	std::shared_ptr<Socket> connect(int timeout);

	/// <summary>
	/// <para>��� �ܸ��� ������ �����Ѵ�.</para>
	/// <para>������ ���������� ����Ǹ� true��, ������ �Ǿ� ���� �ʰų� ������������ ����� ��� false�� ��ȯ�Ѵ�.</para>
	/// </summary>
	/// <returns>���� ���� ���� ����</returns>
	bool close();

	/// <summary>
	/// ���� ���ҽ��� ���� ���� å���� ������ �Ҹ����̴�.
	/// </summary>
	~Socket();

	/// <summary>
	/// <para>Ŭ���̾�Ʈ �ܿ����� ���̴� �Լ��̸�, ������ IP�� ��Ʈ ���� ������ ������ ��� ������ ���� ��ü�� �����Ѵ�.</para>
	/// <para>�־��� ������ ������ �������� ���� ��ü�� ������ �� �ִٸ� ���� Ŭ������ �����͸�, ������ ������ ��� nullptr�� ��ȯ�Ѵ�.</para>
	/// </summary>
	/// <param name="serverIP">���� IP</param>
	/// <param name="serverPort">���� ��Ʈ</param>
	/// <returns>�־��� ������ ������ �������� ���� ��ü�� ������ �� �ִٸ� ���� Ŭ������ ������, ������ ������ ��� nullptr</returns>
	static std::shared_ptr<Socket> create(const std::tstring &serverIP, const std::tstring &serverPort, const bool isTempSocket);


	// ���� recv & send part

	bool sendMSG(std::tstring const t_msg, const ProtocolType protocolType);
	
	bool sendObj(Serializable & obj, const ObjectType objectType);

	bool sendFile(const std::tstring t_path);

	/// <summary>
	/// AsyncTask Work
	/// Ŭ���̾�Ʈ�κ��� ��Ŷ����� �ް� �׿� ���� ������ �Ѵ�.
	/// ��Ŷ����� ���� ����(recv ����)���� �ٽ� ���ο� receiveTo�� �񵿱�� ������ ���� ���� ������ �Ѵ�.
	/// </summary>
	std::shared_ptr<Socket> __receivingLoop();
};