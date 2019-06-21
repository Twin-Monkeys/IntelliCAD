/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: ClientNetwork.cpp
*	작성자			: 오수백
*	최종 수정일		: 19.05.01
*
*	서버와 통신하기 위한 클라이언트 네트워크 최상위 모듈
*/

#include "ClientNetwork.h"
#include "System.h"
#include "MacroTransaction.h"
#include "Debugger.h"
#include "PacketHeader.h"
#include "NetworkStream.h"


using namespace std;

ClientNetwork ClientNetwork::__instance = ClientNetwork();

shared_ptr<Socket> ClientNetwork::getSocket(const bool isRecevingLoop)
{

	if (!isRecevingLoop) { // loop를 돌지 않고, 생성되는 새로운 Socket으로 동기처리 하겠다.
		shared_ptr<Socket> sock = Socket::create(__serverIP, __serverPort, true);
		shared_ptr<Socket> connectedSocket = sock->connect(0);
		return connectedSocket;
	}

	// loop 돌고 비동기 처리 위해 tempSockets에 저장
	if (__pSocket->isReceving() || __pSocket->isSending()) {
		shared_ptr<Socket> sock = Socket::create(__serverIP, __serverPort, true);
		__tempSockets.emplace(sock);
		shared_ptr<Socket> connectedSocket = sock->connect(0);
		System::getSystemContents().getTaskManager().run(TaskType::CONNECTION_CLOSED, *sock, &Socket::__receivingLoop);

		return connectedSocket;
	}
	else return __pSocket;
}

ClientNetwork::ClientNetwork()
{
	//socket version 2.2
	WSAStartup(MAKEWORD(2, 2), &__wsaData);
}

const tstring& ClientNetwork::getServerIP() const
{
	return __serverIP;
}

const tstring& ClientNetwork::getServerPort() const
{
	return __serverPort;
}

bool ClientNetwork::isCreated() const
{
	IF_T_RET_F(__pSocket == nullptr);

	return true;
}

bool ClientNetwork::isConnected() const
{
	IF_T_RET_F(!isCreated());

	IF_T_RET_T(__pSocket->isConnected());

	return false;
}

bool ClientNetwork::createClientSocket(const tstring &serverIP, const tstring &serverPort)
{
	__serverIP = serverIP;
	__serverPort = serverPort;

	__pSocket = Socket::create(serverIP, serverPort, false);

	IF_T_RET_F(!isCreated());

	return true;
}

void ClientNetwork::onServerConnectionResult(std::shared_ptr<Socket> pSocket)
{
	if (pSocket == nullptr)
		return;
	//__pSocket = pSocket;
	System::getSystemContents().getTaskManager().run(TaskType::CONNECTION_CLOSED, *pSocket, &Socket::__receivingLoop);
	__isRun = false;
}

void ClientNetwork::onSystemInit()
{
	EventBroadcaster &eventBroadcaster =
		System::getSystemContents().getEventBroadcaster();

	eventBroadcaster.addServerConnectingListener(*this);
	eventBroadcaster.addConnectionCheckListener(*this);
	eventBroadcaster.addConnectionClosedListener(*this);

	// 세인 추가
	eventBroadcaster.addSystemDestroyListener(*this);
}

void ClientNetwork::onSystemDestroy()
{
	// 세인: 시스템 종료 시 호출
	close();
}

// 세인 추가
void ClientNetwork::onConnectionCheck()
{
	// 서버와 연결 이후 이후 활성화되는 connection check 버튼을 클릭하면 ConnectionCheck 이벤트가 발생함
	// 이곳에서 패킷 send / receive를 통해 서버와 연결 체크를 해볼 것
	
	//__pSocket->sendMSG("Hello Server", ProtocolType::CONNECTION_CHECK);
	sendFile(_T("c:\\network_test\\dummy.txt"));
	sendMSG(_T("Hello Server"), ProtocolType::CONNECTION_CHECK);

	// 이 함수는 "onConnectionCheck"이라는 메세지 박스를 팝업함
	Debugger::popMessageBox(_T("onConnectionCheck"));
}

void ClientNetwork::onConnectionClosed(std::shared_ptr<Socket> pSocket)
{
	
	if (pSocket == __pSocket) __pSocket = nullptr;
	else __tempSockets.erase(pSocket);
}

bool ClientNetwork::connect(const int timeout)
{
	IF_T_RET_F(!isCreated()); // 클라이언트소켓 미생성 상태 false
	IF_T_RET_F(__isRun); // 이미 running 중이면 false

	__isRun = true;

	System::getSystemContents().getTaskManager()
		.run(TaskType::SERVER_CONNECTED, *__pSocket, &Socket::connect, timeout);

	return true;
}

bool ClientNetwork::connectBlocking()
{
	__pSocket = __pSocket->connect(500);
	if (__pSocket == nullptr) return false;

	System::getSystemContents().getTaskManager().run(TaskType::CONNECTION_CLOSED, *__pSocket, &Socket::__receivingLoop);
	return true;
}

bool ClientNetwork::close()
{
	if (__pSocket)
		__pSocket->close();

	for (std::shared_ptr<Socket> sock : __tempSockets) {
		sock->close();
	}
	return true;
}

ClientNetwork& ClientNetwork::getInstance()
{
	return __instance;
}

ClientNetwork::~ClientNetwork()
{
	WSACleanup();
}

void ClientNetwork::sendMSG(std::tstring const msg, const ProtocolType protocolType)
{
	shared_ptr<Socket> sock = getSocket(true);
	
	//GENERIC에서 통신완료 타입으로 바꾸고, listener를 통해 tempSockets에서 제외.
	System::getSystemContents().getTaskManager()
		.run(TaskType::GENERIC, *sock, &Socket::sendMSG, msg, protocolType);
}

void ClientNetwork::sendObj(Serializable & obj, const ObjectType objectType)
{
	shared_ptr<Socket> sock = getSocket(true);

	//GENERIC에서 통신완료 타입으로 바꾸고, listener를 통해 tempSockets에서 제외.
	System::getSystemContents().getTaskManager()
		.run(TaskType::GENERIC, *sock, &Socket::sendObj, obj, objectType);

}

void ClientNetwork::sendFile(const std::tstring path) 
{
	shared_ptr<Socket> sock = getSocket(true);

	//GENERIC에서 통신완료 타입으로 바꾸고, listener를 통해 tempSockets에서 제외.
	System::getSystemContents().getTaskManager()
		.run(TaskType::GENERIC, *sock, &Socket::sendFile, path);

}

AuthorizingResult ClientNetwork::loginRequest(Account &userInfo, const tstring & id, const tstring & password)
{
	shared_ptr<Socket> sock = getSocket(false);

	// id
	sock->sendMSG(id, ProtocolType::LOGIN);
	const PacketHeader * ph = sock->getHeader();
	if (ph->getProtocolType() == ProtocolType::NOK)
		return AuthorizingResult::FAILED_INVALID_ID;
	else if (ph->getProtocolType() == ProtocolType::DB_ERROR)
		return AuthorizingResult::FAILED_DB_ERROR;

	// password
	sock->sendMSG(password, ProtocolType::LOGIN);
	ph = sock->getHeader();
	if (ph->getProtocolType() == ProtocolType::NOK)
		return AuthorizingResult::FAILED_WRONG_PASSWORD;

	// UserInfo recv
	ph = sock->getHeader();
	NetworkStream nStream = NetworkStream(sock, ph->getByteCount());
	userInfo = Account(nStream);

	return AuthorizingResult::SUCCESS;
}

bool ClientNetwork::requestServerDBFile(const ServerFileDBSectionType sectionType, const tstring & name)
{
	shared_ptr<Socket> sock = getSocket(false); // 동기 처리 위한 임시 소켓

	// Send PH
	// Send name
	sock->sendMSG(name, ProtocolType::FILE_REQUEST);
	
	// Send ServerDBSectionType
	memcpy(sock->bufForTemp, &sectionType, sizeof(sectionType));
	sock->send(sock->bufForTemp, sock->BUF_SIZE);

	// Recv EXISTENT (OK, NOK)
	const PacketHeader * ph = sock->getHeader();
	const bool EXISTENT = ph->getProtocolType() == ProtocolType::OK;

	// File Receving Loop Start
	sock->setFilePath(_T("db\\") + name);
	__tempSockets.emplace(sock);
	System::getSystemContents().getTaskManager()
		.run(TaskType::CONNECTION_CLOSED, *sock, &Socket::__receivingLoop);
	
	return EXISTENT;
}

