#pragma once

#include "ReadStream.hpp"
#include "Socket.h"

class NetworkStream : public ReadStream
{
private:
	std::shared_ptr<Socket> sock;

	int size;

public:

	~NetworkStream();
	NetworkStream(std::shared_ptr<Socket> socket, int byteCount);

	virtual bool get(ubyte &buffer) override;
	virtual int get(void *pBuffer, int bufferSize) override;

	virtual int getStreamSize() const override;

};

