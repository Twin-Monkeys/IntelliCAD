/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: Socket.cpp
*	작성자			: 오수백
*	최종 수정일		: 19.05.01
*
*	단대단 통신을 위한 소켓 모듈
*/

#include "stdafx.h"
#include <ws2tcpip.h>
#include "Socket.h"
#include "Parser.hpp"
#include "MacroTransaction.h"
#include "System.h"
#include "FileStream.h"
#include <iostream>
#include "NetworkStream.h"

using namespace std;

Socket::Socket(const SOCKET &hSockRaw, const SOCKADDR_IN &sockAddr, const bool connected)
	: __sock(hSockRaw), __sockAddr(sockAddr), __connected(connected)
{
	BOOL opt = TRUE;
	setsockopt(__sock, IPPROTO_TCP, TCP_NODELAY, (const char*)&opt, sizeof(opt));

}

bool Socket::isConnected() const
{
	return __connected;
}

shared_ptr<Socket> Socket::connect(const int timeout)
{
	cout << "Socket::connect()" << endl;

	//timeout 추가 필요

	//connect
	int err = ::connect(__sock, (SOCKADDR*)&__sockAddr, sizeof(__sockAddr));
	perror("connect error");
	if (err == -1) {
		return nullptr;
	}

	cout << "Socket::connect() success" << endl;

	return shared_from_this();
}

bool Socket::close()
{
	cout << "Socket::close()" << endl;
	IF_T_RET_F(closesocket(__sock) == -1);

	return true;
}

Socket::~Socket()
{
	cout << "Socket::~Socket()" << endl;

	close();
}

shared_ptr<Socket> Socket::create(const tstring &serverIP, const tstring &serverPort)
{
	cout << "Socket::create()" << endl;

	
	string ip = Parser::tstring$string(serverIP);
	const char* ipC = ip.c_str();
	int port = Parser::tstring$Int(serverPort);

	cout << "ip : " << ipC << endl;
	cout << "port : " << port << endl;

	//create socket
	SOCKET sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
	cout << "sock : " << sock << endl;

	//socket structure
	SOCKADDR_IN sockAddr = {};
	sockAddr.sin_family = AF_INET;
	sockAddr.sin_port = htons(port);
	// sockAddr.sin_addr.s_addr = inet_addr(ipC); 
	sockAddr.sin_addr = Parser::ipString$sin_addr(Parser::LPCSTR$tstring(ipC));

	if(sock == INVALID_SOCKET)
		return nullptr;

	return make_shared<Socket>(sock, sockAddr, true);
}

// 이하 send & recv 관련

int Socket::send(const char* const data, const size_t size)
{
	return ::send(__sock, data, static_cast<int>(size), 0);
}

int Socket::recv(char * const p, const size_t size)
{
	return ::recv(__sock, p, BUF_SIZE, 0);
}

const char* Socket::recvMSG()
{
	recv(bufForTemp, BUF_SIZE);
	return bufForTemp;
}

bool Socket::recvObj(const ObjectType objectType, const uint32_t byteCount)
{
	EventBroadcaster eventBroadcaster = System::getSystemContents().getEventBroadcaster();
	NetworkStream nStream = NetworkStream(shared_from_this(), byteCount);

	//임시로 테스트 위해 여기서 add listener
	// eventBroadcaster.addDummyReceivedListener(SuperDummy::getInstance());

	switch (objectType) {
	
	case ObjectType::DUMMY_CLASS:
		// NetworkStream을 통한 DummyClass 객체 생성
		// notify
		cout << "object notify before" << endl;
		// eventBroadcaster.notifyDummyReceived(make_shared<DummyClass>(nStream));
		cout << "object notify after" << endl;
		break;
	}
	
	return true;
}

bool Socket::recvFile(const std::string & path, const uint32_t byteCount)
{
	ofstream fout(path, ios::binary | ios::out);
	
	uint32_t recvCnt, offset = 0, needLength;
	// size_t err;
	memset(bufForTemp, 0, BUF_SIZE);

	while (offset < byteCount) {
		needLength = BUF_SIZE > byteCount - offset ? byteCount - offset : BUF_SIZE;

		recvCnt = recv(bufForTemp, needLength);
		if (recvCnt <= 0) return false;

		fout.write(bufForTemp, recvCnt);
		offset += recvCnt;
	}

	fout.close();

	cout << endl << "Socket::recvFile() recv byte : " << offset << endl;

	return true;
}

const PacketHeader* Socket::getHeader()
{
	int err = recv(bufForTemp, BUF_SIZE);
	if (err <= 0) return nullptr;

	memcpy(bufForPH, bufForTemp, 12);
	return reinterpret_cast<PacketHeader*>(bufForPH);
}

bool Socket::sendHeader(PacketHeader ph)
{
	memcpy(bufForTemp, &ph, sizeof(ph));
	if (send(bufForTemp, BUF_SIZE) == -1) {
		return false;
	}
	return true;
}

bool Socket::isReceving() const
{
	return __isReceving;
}

bool Socket::isSending() const
{
	return __isSending;
}

std::shared_ptr<Socket> Socket::sendMSG(const char * const msg, const ProtocolType protocolType)
{
	__isSending = true;

	PacketHeader ph = PacketHeader(protocolType, ObjectType::MSG, static_cast<uint32_t>(strlen(msg) + 1));
	if (!sendHeader(ph)) return shared_from_this();

	memcpy(bufForTemp, msg, strlen(msg) + 1);
	if (send(bufForTemp, BUF_SIZE) == -1) {
		__isSending = false;
		return shared_from_this();
	}

	__isSending = false;

	return shared_from_this();
}

std::shared_ptr<Socket> Socket::sendObj(Serializable & obj, const ObjectType objectType)
{
	__isSending = true;

	uint32_t readCnt, offset = 0, sendLength;
	int elemOffset;

	uint32_t byteCount = obj.getStream().getStreamSize();
	std::vector<ElementMeta> elements = obj._getStreamMeta();
	size_t err;
	memset(bufForTemp, 0, BUF_SIZE);


	//send packet header
	PacketHeader ph = PacketHeader(ProtocolType::OBJECT, objectType, byteCount);
	sendHeader(ph);

	//send object
	for (const ElementMeta & em : elements) { // 멤버 변수 별 전송 

		if (em.elemSize < BUF_SIZE) {
			readCnt = obj.serialAccess(offset, reinterpret_cast<ubyte *>(bufForTemp), em.elemSize);
			err = send(bufForTemp, BUF_SIZE); // BUF_SIZE 보다 작은 경우, BUF_SIZE로 전송
			if (err == -1) {
				__isSending = false;
				return shared_from_this();
			}
			offset += readCnt;
			continue; // 보냈으므로 다음 element 전송
		}

		// BUF_SIZE보다 큰 경우는 분할 전송
		elemOffset = 0;
		while (elemOffset < em.elemSize) {
			sendLength = BUF_SIZE > em.elemSize - elemOffset ? em.elemSize - elemOffset : BUF_SIZE;

			readCnt = obj.serialAccess(offset, reinterpret_cast<ubyte *>(bufForTemp), sendLength); // ObjectStream.get()으로 치환가능
			err = send(bufForTemp, readCnt);
			if (err == -1) {
				__isSending = false;
				return shared_from_this();
			}
			elemOffset += readCnt;
			offset += readCnt;
		}
	}

	__isSending = false;

	return shared_from_this();
}

std::shared_ptr<Socket> Socket::sendFile(const std::string & path)
{
	__isSending = true;


	FileStream fstream = FileStream(path);
	uint32_t readCnt, offset = 0;
	uint32_t byteCount = fstream.getStreamSize();
	size_t err;
	memset(bufForTemp, 0, BUF_SIZE);
	
	// send packet header
	PacketHeader ph = PacketHeader(ProtocolType::FILE, ObjectType::MAX, byteCount);
	sendHeader(ph);

	Sleep(10000);

	// send file name

	// send file content
	while (offset < byteCount) {
		readCnt = fstream.get(bufForTemp, BUF_SIZE);
		err = send(bufForTemp, readCnt);
		if (err == -1) {
			__isSending = false;
			return shared_from_this();
		}
		offset += readCnt;
	}

	cout << endl << "Socket::sendFile() send byte : " << offset << endl;

	__isSending = false;

	return shared_from_this();
}

void Socket::__receivingLoop()
{
	while (true) {
		__isReceving = false;

		//recv packet header
		const PacketHeader* const ph = getHeader();

		__isReceving = true; // running 중

		if (ph == nullptr) {
			close();
			return;
			//ClientNetwork의 __pSocket을 nullptr로 초기화
		}

		cout << endl
			<< "recv ByteCount : " << ph->getByteCount() << " "
			<< "recv ProtocolType : " << (int)ph->getProtocolType() << " "
			<< "recv ObjectType : " << (int)ph->getObjectType()
			<< endl;

		// protocol processing(recv content 포함)
		switch (ph->getProtocolType()) {

		case ProtocolType::CONNECTION_CHECK:
			cout << endl << "ProtocolType::CONNECTION_CHECK" << endl;
			cout << "recvMSG() : " << recvMSG() << endl;
			sendMSG("Hello Response : Hello Server", ProtocolType::CONNECTION_RESPONSE);
			break;

		case ProtocolType::CONNECTION_RESPONSE:
			cout << endl << "ProtocolType::CONNECTION_RESPONSE" << endl;
			cout << "recvMSG() : " << recvMSG() << endl;
			break;

		case ProtocolType::OBJECT:
			cout << endl << "ProtocolType::OBJECT" << endl;
			recvObj(ph->getObjectType(), ph->getByteCount());
			break;
		case ProtocolType::FILE:
			cout << endl << "ProtocolType::FILE" << endl;
			recvFile(string("c:\\network_test\\received\\dummy.txt"), ph->getByteCount()); // 경로? 파일명? 확장자?
			break;
		}

	}

}