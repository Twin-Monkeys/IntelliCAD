#pragma once

#include <memory>
#include "TransmittingDirectionType.h"

class StreamTransmittingListener
{
public:

	// who: ��Ʈ�� �ۼ��� ���� ����
	// directionType: ���� ���� (up: ���ʿ��� ���� ������ ��) / (down: ��밡 �������� ������ ��)
	// transmittedSize: ������ ������ �� (���� �߿� ������ ����� �� ���̰�, ���� �������� ¥���� ũ�Ⱑ �� ����)
	virtual void onStreamTransmitting(
		std::shared_ptr<const class Socket> who,
		TransmittingDirectionType directionType, const int transmittedSize) = 0;
};