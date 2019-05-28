#include "NetworkStream.h"
#include <iostream>

NetworkStream::NetworkStream(std::shared_ptr<Socket> socket, int byteCount)
	: sock(socket), size(byteCount)
{
}

bool NetworkStream::get(ubyte & buffer)
{
	return sock->recv(reinterpret_cast<char *>(&buffer), 1);
}

int NetworkStream::get(void * pBuffer, int bufferSize)
{
	// BUF_SIZE���� ���� ���� BUF_SIZE�� ���۵ǹǷ� BUF_SIZE�� ����
	if (bufferSize < sock->BUF_SIZE) {
		sock->recv(sock->bufForTemp, sock->BUF_SIZE);
		memcpy(pBuffer, sock->bufForTemp, bufferSize);
		return bufferSize;
	}

	// BUF_SIZE���� ū ���� ���� �����̹Ƿ� ���� ����
	int offset = 0, needLength, readCnt;
	while (offset < bufferSize) {
		needLength = sock->BUF_SIZE > bufferSize - offset ? bufferSize - offset : sock->BUF_SIZE;

		readCnt = sock->recv(reinterpret_cast<char *>(pBuffer) + offset, needLength);
		if (readCnt <= 0) return -1;

		offset += readCnt;
	}

	return offset;
}


int NetworkStream::getStreamSize() const
{
	return size;
}


NetworkStream::~NetworkStream()
{
}
