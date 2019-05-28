#pragma once
#include <memory>
#include "Socket.h"

class ConnectionClosedListener
{
public:
	virtual void onConnectionClosed(std::shared_ptr<Socket> pSocket) = 0;
};