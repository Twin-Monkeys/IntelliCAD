/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: ClientNetwork.h
*	작성자			: 오수백
*	최종 수정일		: 19.05.01
*
*	서버와 통신하기 위한 클라이언트 네트워크 최상위 모듈
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
#include "AuthorizingResult.h"
#include "Account.h"
#include "SystemDestroyListener.h"
#include "ServerFileDBSectionType.h"

/// <summary>
/// <para>서버와 통신하기 위한 클라이언트 네트워크 모듈이다.</para>
/// <para>ClientNetwork 객체는 응용 내에서 오직 하나의 객체만 존재할 수 있다.</para>
/// <para>유일 객체는 <c>ClientNetwork::getInstance()</c> 함수를 통해 얻는다.</para>
/// <para>서버와 연결하기 위해서는 서버와 통신하기 위한 <c>Socket</c> 객체를 먼저 생성해야 한다.</para>
/// <para><c>Socket</c> 객체를 생성하려면 <c>ClientNetwork::createClientSocket()</c> 함수를 이용한다.</para>
/// <para><c>Socket</c> 객체가 생성 되었다면 <c>ClientNetwork::connect()</c> 함수를 통해 서버와 연결한다.</para>
/// </summary>
class ClientNetwork : 
	public SystemInitListener, public SystemDestroyListener, public ServerConnectingListener,
	public ConnectionCheckListener, public ConnectionClosedListener
{
private:
	WSADATA __wsaData;

	bool __isRun = false;

	/// <summary>
	/// 응용 내에 존재하는 유일 객체이다.
	/// </summary>
	static ClientNetwork __instance;

	std::tstring __serverIP = _T("127.0.0.1");
	std::tstring __serverPort = _T("9000");

	/// <summary>
	/// 항상 서버와 연결되어있는 소켓
	/// </summary>
	std::shared_ptr<Socket> __pSocket = nullptr;

	// 다중 처리 위한 멤버변수, 메서드	
	std::set<std::shared_ptr<Socket>> __tempSockets;
	std::shared_ptr<Socket> getSocket(const bool isRecevingLoop);

	/// <summary>
	/// <para>유일 객체를 생성하기 위한 내부용 기본 생성자.</para>
	/// <para>네트워크 통신과 관련한 초기화 과정이 포함된다.</para>
	/// </summary>
	ClientNetwork();

	ClientNetwork(const ClientNetwork &) = delete;
	ClientNetwork(ClientNetwork &&) = delete;

public:
	/// <summary>
	/// <para>마지막으로 연결된 서버의 IP 값을 반환한다.</para>
	/// <para>한번도 연결된 적이 없는 경우 0.0.0.0을 반환한다.</para>
	/// </summary>
	/// <returns>마지막으로 연결된 서버의 IP 값</returns>
	const std::tstring& getServerIP() const;

	/// <summary>
	/// <para>마지막으로 연결된 서버의 포트 값을 반환한다.</para>
	/// <para>한번도 연결된 적이 없는 경우 9000을 반환한다.</para>
	/// </summary>
	/// <returns>마지막으로 연결된 서버의 포트 값</returns>
	const std::tstring& getServerPort() const;

	/// <summary>
	/// <para><c>ClientSocket</c> 객체가 생성되어 있는지 여부를 반환한다.</para>
	/// <para><c>ClientSocket</c> 객체는 <c>ClientNetwork::createClientSocket()</c> 함수를 통해 얻을 수 있다.</para>
	/// </summary>
	/// <returns><c>ClientSocket</c> 객체가 생성되어 있는지 여부</returns>
	bool isCreated() const;

	/// <summary>
	/// <para>서버와 연결이 되어있는 상태인지 여부를 반환한다.</para>
	/// <para>서버와의 연결은 <c>Socket</c> 객체를 생성한 뒤 <c>ClientNetwork::connect()</c> 함수를 이용한다.</para>
	/// </summary>
	/// <returns>서버 연결 여부</returns>
	bool isConnected() const;

	/// <summary>
	/// <para>서버와 통신하기 위한 <c>Socket</c> 객체를 생성한다.</para>
	/// <para>인자로는 서버의 IP와 포트가 필요하다.</para>
	/// <para>객체 생성 성공시 true를, 여러가지 이유로 인한 실패 시 false를 반환한다.</para>
	/// <para>실제 서버와 통신하기 위해서는 <c>ClientNetwork::connect()</c> 함수를 통한 별도의 연결 작업이 필요하다.</para>
	/// </summary>
	/// <param name="serverIP">서버의 IP</param>
	/// <param name="serverPort">서버의 포트</param>
	/// <returns><c>Socket</c> 객체 생성 결과</returns>
	bool createClientSocket(const std::tstring &serverIP, const std::tstring &serverPort);

	virtual void onServerConnectionResult(std::shared_ptr<Socket> pSocket) override;
	virtual void onSystemInit() override;
	virtual void onSystemDestroy() override;
	virtual void onConnectionCheck() override;
	virtual void onConnectionClosed(std::shared_ptr<Socket> pSocket) override;

	/// <summary>
	/// <para><c>ClientNetwork::createClientSocket()</c> 함수를 이용하여 <c>Socket</c> 객체를 생성한 뒤, 서버와 연결을 시도한다.</para>
	/// <para>연결 성공 시 true를, 인자로 주어진 제한 시간 내에 서버와 연결을 수립할 수 없으면 false를 반환한다.</para>
	/// <para>제한 시간은 밀리세컨드 단위이다.</para>
	/// </summary>
	/// <param name="timeout">연결 수립 시도 후 최대 대기 시간</param>
	/// <returns>서버와 연결 성공 여부</returns>
	bool connect(int timeout = 5000);

	bool connectBlocking();

	/// <summary>
	/// <para>서버와의 연결을 종료한다.</para>
	/// <para>정상적으로 종료가 될 시 true를, 연결되어 있지 않은 상태이거나 비정상적으로 종료되면 false를 반환한다.</para>
	/// </summary>
	/// <returns>연결의 정상 종료 여부</returns>
	bool close();

	/// <summary>
	/// ClientNetwork의 유일 객체를 반환한다.
	/// </summary>
	/// <returns>ClientNetwork 객체</returns>
	static ClientNetwork& getInstance();

	/// <summary>
	/// <para>소멸자이다.</para>
	/// <para>네트워크 통신과 관련한 릴리즈 과정이 포함된다.</para>
	/// </summary>
	~ClientNetwork();

	// 비동기 send
	void sendMSG(std::tstring const msg, const ProtocolType protocolType);

	void sendObj(Serializable & obj, const ObjectType objectType);

	void sendFile(const std::tstring path);

	AuthorizingResult loginRequest(Account &account, const std::tstring & id, const std::tstring & password);

	bool requestServerDBFile(const ServerFileDBSectionType sectionType, const std::tstring &name);
};