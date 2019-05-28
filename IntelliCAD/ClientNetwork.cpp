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
#include <iostream>

using namespace std;

ClientNetwork ClientNetwork::__instance = ClientNetwork();

shared_ptr<Socket> ClientNetwork::getSocket()
{
	cout << "__pSocket->isReceving() : " << __pSocket->isReceving();
	cout << "__pSocket->isSending() : " << __pSocket->isSending();

	if (__pSocket->isReceving() || __pSocket->isSending()) {
		shared_ptr<Socket> sock = Socket::create(__serverIP, __serverPort);
		__tempSockets.emplace(sock);
		shared_ptr<Socket> connectedSocket = sock->connect(0);

		System::getSystemContents().getTaskManager().
			run_void(TaskType::GENERIC, *sock, &Socket::__receivingLoop);

		cout << "getSocket() return tempSocket" << endl;

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

	__pSocket = Socket::create(serverIP, serverPort);

	IF_T_RET_F(!isCreated());

	cout << "socket create success" << endl;

	return true;
}

void ClientNetwork::onServerConnectionResult(std::shared_ptr<Socket> pSocket)
{
	if (pSocket == nullptr) {
		cout << "server connect fail. please retry." << endl;
		cout << "ClientNetwork::onServerConnectionResult() : pSocket == nullptr" << endl;
		
		__pSocket->close();
		__pSocket = nullptr;
		__isRun = false;

		return;
	}
	//__pSocket = pSocket;
	System::getSystemContents().getTaskManager().
		run_void(TaskType::GENERIC, *pSocket, &Socket::__receivingLoop);

	__isRun = false;
}

void ClientNetwork::onSystemInit()
{
	EventBroadcaster &eventBroadcaster =
		System::getSystemContents().getEventBroadcaster();

	eventBroadcaster.addServerConnectingListener(*this);
	eventBroadcaster.addConnectionCheckListener(*this);
	eventBroadcaster.addConnectionClosedListener(*this);
}

// 세인 추가
void ClientNetwork::onConnectionCheck()
{
	// 서버와 연결 이후 이후 활성화되는 connection check 버튼을 클릭하면 ConnectionCheck 이벤트가 발생함
	// 이곳에서 패킷 send / receive를 통해 서버와 연결 체크를 해볼 것
	
	//__pSocket->sendMSG("Hello Server", ProtocolType::CONNECTION_CHECK);
	
	// sendFile(*(new string("c:\\network_test\\dummy.txt")));	// 메모리 누수
	sendFile("c:\\network_test\\dummy.txt");

	/*char * msg = new char[20]; 
	strcpy(msg, "Hello Server");*/ // 메모리 누수
	char msg[] = "Hello Server";
	sendMSG(msg, ProtocolType::CONNECTION_CHECK);

	// 이 함수는 "onConnectionCheck"이라는 메세지 박스를 팝업함
	Debugger::popMessageBox(_T("onConnectionCheck"));
}

void ClientNetwork::onConnectionClosed(std::shared_ptr<Socket> pSocket)
{
	cout << "ClientNetwork::onConnectionClosed()" << endl;
	
	set<shared_ptr<Socket>> erase;
	for (shared_ptr<Socket> sock : __taskCompletedSockets) {
		if (!sock->isSending() && !sock->isReceving()) {
			__tempSockets.erase(sock);
			erase.emplace(sock);
		}
	}
	for (shared_ptr<Socket> sock : erase) {
		__taskCompletedSockets.erase(sock);
	}

	if (pSocket == __pSocket) return;
	else __taskCompletedSockets.emplace(pSocket);
}

bool ClientNetwork::connect(const int timeout)
{
	IF_T_RET_F(!isCreated()); // 클라이언트소켓 미생성 상태 false
	IF_T_RET_F(__isRun); // 이미 running 중이면 false

	__isRun = true;

	System::getSystemContents().getTaskManager()
		.run(TaskType::SERVER_CONNECTED, *__pSocket, &Socket::connect, timeout);

	//2개 test
//	__testSocket = Socket::create(_T("127.0.0.1"), _T("9000"));
//	System::getInstance().taskMgr
//		.run(TaskType::SERVER_CONNECTED, *__testSocket, &Socket::connect, timeout);

	return true;
}

bool ClientNetwork::close()
{
	__pSocket = nullptr;
	return true;
}

ClientNetwork& ClientNetwork::getInstance()
{
	return __instance;
}

ClientNetwork::~ClientNetwork()
{
	close();
	WSACleanup();
}

void ClientNetwork::sendMSG(const char * const msg, const ProtocolType protocolType)
{
	shared_ptr<Socket> sock = getSocket();
	
	//GENERIC에서 통신완료 타입으로 바꾸고, listener를 통해 tempSockets에서 제외.
	System::getSystemContents().getTaskManager()
		.run(TaskType::CONNECTION_CLOSED, *sock, &Socket::sendMSG, msg, protocolType);
}

void ClientNetwork::sendObj(Serializable & obj, const ObjectType objectType)
{
	shared_ptr<Socket> sock = getSocket();

	//GENERIC에서 통신완료 타입으로 바꾸고, listener를 통해 tempSockets에서 제외.
	System::getSystemContents().getTaskManager()
		.run(TaskType::CONNECTION_CLOSED, *sock, &Socket::sendObj, obj, objectType);
}

void ClientNetwork::sendFile(const std::string & path) 
{
	shared_ptr<Socket> sock = getSocket();

	//GENERIC에서 통신완료 타입으로 바꾸고, listener를 통해 tempSockets에서 제외.
	System::getSystemContents().getTaskManager()
		.run(TaskType::CONNECTION_CLOSED, *sock, &Socket::sendFile, path);
}