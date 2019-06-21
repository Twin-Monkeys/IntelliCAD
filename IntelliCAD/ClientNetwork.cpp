/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	���ϸ�			: ClientNetwork.cpp
*	�ۼ���			: ������
*	���� ������		: 19.05.01
*
*	������ ����ϱ� ���� Ŭ���̾�Ʈ ��Ʈ��ũ �ֻ��� ���
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

	if (!isRecevingLoop) { // loop�� ���� �ʰ�, �����Ǵ� ���ο� Socket���� ����ó�� �ϰڴ�.
		shared_ptr<Socket> sock = Socket::create(__serverIP, __serverPort, true);
		shared_ptr<Socket> connectedSocket = sock->connect(0);
		return connectedSocket;
	}

	// loop ���� �񵿱� ó�� ���� tempSockets�� ����
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

	// ���� �߰�
	eventBroadcaster.addSystemDestroyListener(*this);
}

void ClientNetwork::onSystemDestroy()
{
	// ����: �ý��� ���� �� ȣ��
	close();
}

// ���� �߰�
void ClientNetwork::onConnectionCheck()
{
	// ������ ���� ���� ���� Ȱ��ȭ�Ǵ� connection check ��ư�� Ŭ���ϸ� ConnectionCheck �̺�Ʈ�� �߻���
	// �̰����� ��Ŷ send / receive�� ���� ������ ���� üũ�� �غ� ��
	
	//__pSocket->sendMSG("Hello Server", ProtocolType::CONNECTION_CHECK);
	sendFile(_T("c:\\network_test\\dummy.txt"));
	sendMSG(_T("Hello Server"), ProtocolType::CONNECTION_CHECK);

	// �� �Լ��� "onConnectionCheck"�̶�� �޼��� �ڽ��� �˾���
	Debugger::popMessageBox(_T("onConnectionCheck"));
}

void ClientNetwork::onConnectionClosed(std::shared_ptr<Socket> pSocket)
{
	
	if (pSocket == __pSocket) __pSocket = nullptr;
	else __tempSockets.erase(pSocket);
}

bool ClientNetwork::connect(const int timeout)
{
	IF_T_RET_F(!isCreated()); // Ŭ���̾�Ʈ���� �̻��� ���� false
	IF_T_RET_F(__isRun); // �̹� running ���̸� false

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
	
	//GENERIC���� ��ſϷ� Ÿ������ �ٲٰ�, listener�� ���� tempSockets���� ����.
	System::getSystemContents().getTaskManager()
		.run(TaskType::GENERIC, *sock, &Socket::sendMSG, msg, protocolType);
}

void ClientNetwork::sendObj(Serializable & obj, const ObjectType objectType)
{
	shared_ptr<Socket> sock = getSocket(true);

	//GENERIC���� ��ſϷ� Ÿ������ �ٲٰ�, listener�� ���� tempSockets���� ����.
	System::getSystemContents().getTaskManager()
		.run(TaskType::GENERIC, *sock, &Socket::sendObj, obj, objectType);

}

void ClientNetwork::sendFile(const std::tstring path) 
{
	shared_ptr<Socket> sock = getSocket(true);

	//GENERIC���� ��ſϷ� Ÿ������ �ٲٰ�, listener�� ���� tempSockets���� ����.
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
	shared_ptr<Socket> sock = getSocket(false); // ���� ó�� ���� �ӽ� ����

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

