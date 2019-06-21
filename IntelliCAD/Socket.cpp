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
#include "NetworkStream.h"
#include "Debugger.h"

using namespace std;

Socket::Socket(const SOCKET &hSockRaw, const SOCKADDR_IN &sockAddr, const bool connected, const bool isTempSocket)
	: __sock(hSockRaw), __sockAddr(sockAddr), __connected(connected), __isTempSocket(isTempSocket)
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
	int err = ::connect(__sock, (SOCKADDR*)&__sockAddr, sizeof(__sockAddr));
	if (err == -1) {
		return nullptr;
	}

	__connected = true;
	return shared_from_this();
}

bool Socket::close()
{
	IF_T_RET_F(closesocket(__sock) == -1);

	return true;
}

Socket::~Socket()
{
	close();
}

shared_ptr<Socket> Socket::create(const tstring &serverIP, const tstring &serverPort, const bool isTempSocket)
{
	int port = Parser::tstring$int(serverPort);

	//create socket
	SOCKET sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);

	//socket structure
	SOCKADDR_IN sockAddr = {};
	sockAddr.sin_family = AF_INET;
	sockAddr.sin_port = htons(port);
	sockAddr.sin_addr = Parser::ipString$sin_addr(serverIP); //inet_addr(ipC); 

	if(sock == INVALID_SOCKET)
		return nullptr;

	return make_shared<Socket>(sock, sockAddr, false, isTempSocket);
}

// 이하 send & recv 관련

int Socket::send(const char* const data, const size_t size)
{
	return ::send(__sock, data, static_cast<int>(size), 0);
}

int Socket::recv(char * const p, const size_t size)
{
	return ::recv(__sock, p, static_cast<int>(size), 0);
}

std::tstring Socket::recvMSG()
{
	recv(bufForTemp, BUF_SIZE);
	return Parser::LPCSTR$tstring(bufForTemp);
}

bool Socket::recvObj(const ObjectType objectType, const uint32_t byteCount)
{
	EventBroadcaster eventBroadcaster = System::getSystemContents().getEventBroadcaster();
	NetworkStream nStream = NetworkStream(shared_from_this(), byteCount);

	switch (objectType) {
	
	case ObjectType::DUMMY_CLASS:
		// NetworkStream을 통한 DummyClass 객체 생성
		// notify
		
		//eventBroadcaster.notifyDummyReceived(make_shared<DummyClass>(nStream));
		break;
	}
	
	return true;
}

bool Socket::recvFile(const std::tstring t_path, const uint32_t byteCount)
{
	const string path = Parser::tstring$string(t_path);

	//notify
	EventBroadcaster & eb = System::getSystemContents().getEventBroadcaster();

	ofstream fout(path, ios::binary | ios::out);
	
	uint32_t recvCnt, offset = 0, needLength;
	memset(bufForTemp, 0, BUF_SIZE);

	while (offset < byteCount) {
		needLength = BUF_SIZE > byteCount - offset ? byteCount - offset : BUF_SIZE;

		recvCnt = recv(bufForTemp, needLength);
		if (recvCnt <= 0) return false;

		fout.write(bufForTemp, recvCnt);
		offset += recvCnt;

		//notify
		
	}

	fout.close();

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

void Socket::setFilePath(std::tstring & filePath)
{
	__filePath = filePath;
}

bool Socket::isReceving() const
{
	return __isReceving;
}

bool Socket::isSending() const
{
	return __isSending;
}

bool Socket::sendMSG(tstring const t_msg, const ProtocolType protocolType)
{
	__isSending = true;

	string s = Parser::tstring$string(t_msg);
	const char * msg = s.c_str();

	PacketHeader ph = PacketHeader(protocolType, ObjectType::MSG, static_cast<uint32_t>(strlen(msg) + 1));
	if (!sendHeader(ph)) return false;

	memcpy(bufForTemp, msg, strlen(msg) + 1);
	if (send(bufForTemp, BUF_SIZE) == -1) {
		__isSending = false;
		return false;
	}

	__isSending = false;

	return true;
}

bool Socket::sendObj(Serializable & obj, const ObjectType objectType)
{
	__isSending = true;

	uint32_t readCnt, offset = 0, sendLength, elemOffset;
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
				return false;
			}
			offset += readCnt;
			continue; // 보냈으므로 다음 element 전송
		}

		// BUF_SIZE보다 큰 경우는 분할 전송
		elemOffset = 0;
		while (static_cast<int>(elemOffset) < em.elemSize) {
			sendLength = BUF_SIZE > em.elemSize - elemOffset ? em.elemSize - elemOffset : BUF_SIZE;

			readCnt = obj.serialAccess(offset, reinterpret_cast<ubyte *>(bufForTemp), sendLength); // ObjectStream.get()으로 치환가능
			err = send(bufForTemp, readCnt);
			if (err == -1) {
				__isSending = false;
				return false;
			}
			elemOffset += readCnt;
			offset += readCnt;
		}
	}

	__isSending = false;

	return true;
}

bool Socket::sendFile(const std::tstring t_path)
{
	const string path = Parser::tstring$string(t_path);

	__isSending = true;


	FileStream fstream = FileStream(path);
	uint32_t readCnt, offset = 0;
	uint32_t byteCount = fstream.getStreamSize();
	size_t err;
	memset(bufForTemp, 0, BUF_SIZE);
	
	// send packet header
	PacketHeader ph = PacketHeader(ProtocolType::FILE_RESPONSE, ObjectType::MAX, byteCount);
	sendHeader(ph);

	// send file name

	// send file content
	while (offset < byteCount) {
		readCnt = fstream.get(bufForTemp, BUF_SIZE);
		err = send(bufForTemp, readCnt);
		if (err == -1) {
			__isSending = false;
			return false;
		}
		offset += readCnt;
	}

	__isSending = false;

	return true;
}

std::shared_ptr<Socket> Socket::__receivingLoop()
{
	while (true) {
		__isReceving = false;

		//recv packet header
		const PacketHeader* const ph = getHeader();

		__isReceving = true; // running 중

		// 세인: 프로그램 종료 시 이 부분에서 오류 발생
		if (ph == nullptr) {
			return shared_from_this();
		}

		// protocol processing(recv content 포함)
		switch (ph->getProtocolType()) {

		case ProtocolType::CONNECTION_CHECK:
			Debugger::popMessageBox(_T("CONNECTION_CHECK"));
			sendMSG(_T("Hello Response : Hello Server"), ProtocolType::CONNECTION_RESPONSE);
			break;

		case ProtocolType::CONNECTION_RESPONSE:
			Debugger::popMessageBox(recvMSG());
			break;

		case ProtocolType::OBJECT:
			recvObj(ph->getObjectType(), ph->getByteCount());
			break;
		case ProtocolType::FILE_REQUEST:
			//recvFile(string("c:\\network_test\\received\\dummy.txt"), ph->getByteCount()); // 경로? 파일명? 확장자?
			break;
		case ProtocolType::FILE_RESPONSE:
			recvFile(__filePath, ph->getByteCount());
			break;
		case ProtocolType::PROTOCOL_SUCCESS:
			if (__isTempSocket) return shared_from_this(); // 임시 소켓이라면 프로토콜 과정 종료 시 삭제
			break;
		}

	}

}