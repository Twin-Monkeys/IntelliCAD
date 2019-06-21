#pragma once

#include <memory>
#include "TransmittingDirectionType.h"

class StreamTransmissionFinishedListener
{
public:
	// who: ��Ʈ�� �ۼ��� ���� ����
	// directionType: ���� ���� (up: ���ʿ��� ���� ������ ��) / (down: ��밡 �������� ������ ��)
	virtual void onStreamTransmissionFinished(
		std::shared_ptr<const class Socket> who, TransmittingDirectionType directionType) = 0;
};