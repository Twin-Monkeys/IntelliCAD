/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: Socket.h
*	작성자			: 오수백
*	최종 수정일		: 19.05.01
*
*	단대단 통신을 위한 소켓 모듈
*/

#pragma once

#include <WinSock2.h>
#include <memory>
#include "tstring.h"
#include "PacketHeader.h"
#include "Serializable.h"


/// <summary>
/// <para>단대단 통신을 위한 소켓 클래스이다.</para>
/// <para>서버와 클라이언트 양쪽에서 공통적으로 쓰이는 클래스이다.</para>
/// <para>객체의 생성은 생성자를 직접 호출하거나, <c>Socket::create()</c> 함수를 이용한다.</para>
/// <para><c>Socket::create()</c> 함수는 클라이언트 단에서만 쓰이며, 추후 <c>Socket::connect()</c>를 통해 서버와 연결하여야 한다.</para>
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


	//이하 recv & send part

	static const int BUF_SIZE = 1024;

	int send(const char* const data, const size_t size);
	int recv(char* const p, const size_t size);

	char bufForPH[12];
	char bufForTemp[BUF_SIZE];

	std::tstring recvMSG();

	/// <summary>
	/// Serializable 을 상속한 Object를 네트워크로부터 recv한다.
	/// recv한 데이터를 가지고 Object를 생성한다.
	/// 생성한 Object를 notify하여 Object를 관리하는 상위 Listener 탑재 Object에게 전달한다.
	/// notify 시 전달되는 값은 Object의 shared_ptr 이다.
	/// </summary>
	bool recvObj(const ObjectType objectType, const uint32_t byteCount); // notify 필요

	bool recvFile(const std::tstring t_path, const uint32_t byteCount); // notify 필요

	const PacketHeader* getHeader();

	bool sendHeader(PacketHeader ph);

public:
	void setFilePath(std::tstring & filePath);

	bool isReceving() const;
	bool isSending() const;

	/// <summary>
	/// 소켓 클래스의 명시적 생성자이다.
	/// </summary>
	/// <param name="hSockRaw">raw data</param>
	/// <param name="sockAddr">raw data</param>
	/// <param name="connected">상대 단말과의 연결 여부</param>
	Socket(const SOCKET &hSockRaw, const SOCKADDR_IN &sockAddr, bool connected, bool isTempSocket);

	/// <summary>
	/// 현재 객체가 상대 단말과 연결되어 있는지 여부를 반환한다.
	/// </summary>
	/// <returns>상대 단말과 연결되어 있는지 여부</returns>
	bool isConnected() const;

	/// <summary>
	/// <para>상대 단말과 연결을 시도한다.</para>
	/// <para>인자로 연결에 대한 제한 시간을 입력한다.</para>
	/// <para>연결에 성공 시 현재 객체에 대한 포인터를, 연결에 실패하거나 제한 시간을 넘을 시 nullptr를 반환한다.</para>
	/// </summary>
	/// <param name="timeout">연결 시도에 대한 제한시간</param>
	/// <returns>연결에 성공 시 현재 객체에 대한 포인터, 연결에 실패하거나 제한 시간을 넘을 시 nullptr</returns>
	std::shared_ptr<Socket> connect(int timeout);

	/// <summary>
	/// <para>상대 단말과 연결을 종료한다.</para>
	/// <para>연결이 정상적으로 종료되면 true를, 연결이 되어 있지 않거나 비정상적으로 종료된 경우 false를 반환한다.</para>
	/// </summary>
	/// <returns>연결 종료 성공 여부</returns>
	bool close();

	/// <summary>
	/// 소켓 리소스에 대한 해제 책임을 가지는 소멸자이다.
	/// </summary>
	~Socket();

	/// <summary>
	/// <para>클라이언트 단에서만 쓰이는 함수이며, 서버의 IP와 포트 값을 가지고 서버와 통신 가능한 소켓 객체를 생성한다.</para>
	/// <para>주어진 정보를 가지고 정상적인 소켓 객체를 생성할 수 있다면 소켓 클래스의 포인터를, 생성에 실패한 경우 nullptr를 반환한다.</para>
	/// </summary>
	/// <param name="serverIP">서버 IP</param>
	/// <param name="serverPort">서버 포트</param>
	/// <returns>주어진 정보를 가지고 정상적인 소켓 객체를 생성할 수 있다면 소켓 클래스의 포인터, 생성에 실패한 경우 nullptr</returns>
	static std::shared_ptr<Socket> create(const std::tstring &serverIP, const std::tstring &serverPort, const bool isTempSocket);


	// 이하 recv & send part

	bool sendMSG(std::tstring const t_msg, const ProtocolType protocolType);
	
	bool sendObj(Serializable & obj, const ObjectType objectType);

	bool sendFile(const std::tstring t_path);

	/// <summary>
	/// AsyncTask Work
	/// 클라이언트로부터 패킷헤더를 받고 그에 따른 대응을 한다.
	/// 패킷헤더를 받은 시점(recv 직후)에서 다시 새로운 receiveTo를 비동기로 돌리고 나서 위의 대응을 한다.
	/// </summary>
	std::shared_ptr<Socket> __receivingLoop();
};