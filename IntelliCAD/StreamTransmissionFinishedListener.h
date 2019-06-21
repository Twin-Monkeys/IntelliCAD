#pragma once

#include <memory>
#include "TransmittingDirectionType.h"

class StreamTransmissionFinishedListener
{
public:
	// who: 스트림 송수신 중인 소켓
	// directionType: 전송 방향 (up: 내쪽에서 상대로 보내는 중) / (down: 상대가 내쪽으로 보내는 중)
	virtual void onStreamTransmissionFinished(
		std::shared_ptr<const class Socket> who, TransmittingDirectionType directionType) = 0;
};