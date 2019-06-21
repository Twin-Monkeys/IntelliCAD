#pragma once

#include <memory>
#include "TransmittingDirectionType.h"

class StreamTransmittingListener
{
public:

	// who: 스트림 송수신 중인 소켓
	// directionType: 전송 방향 (up: 내쪽에서 상대로 보내는 중) / (down: 상대가 내쪽으로 보내는 중)
	// transmittedSize: 전송한 데이터 양 (전송 중엔 버퍼의 사이즈가 될 것이고, 전송 마지막엔 짜투리 크기가 될 것임)
	virtual void onStreamTransmitting(
		std::shared_ptr<const class Socket> who,
		TransmittingDirectionType directionType, const int transmittedSize) = 0;
};