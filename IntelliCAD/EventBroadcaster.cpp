/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	���ϸ�			: EventBroadcaster.h
*	�ۼ���			: �̼���
*	���� ������		: 19.03.06
*
*	�̺�Ʈ ���Ŀ� Ŭ����
*/

#include "stdafx.h"
#include "EventBroadcaster.h"
#include "MacroTransaction.h"

using namespace std;

bool EventBroadcaster::addGenericListener(GenericListener &listener)
{
	return __genericListeners.emplace(&listener).second;
}

bool EventBroadcaster::addVolumeLoadingListener(VolumeLoadingListener &listener)
{
	return __volumeLoadingListeners.emplace(&listener).second;
}

bool EventBroadcaster::removeGenericListener(GenericListener &listener)
{
	return __genericListeners.erase(&listener);
}

bool EventBroadcaster::removeVolumeLoadingListener(VolumeLoadingListener &listener)
{
	return __volumeLoadingListeners.erase(&listener);
}

void EventBroadcaster::notifyGeneric() const
{
	for (GenericListener *const pListener : __genericListeners)
		pListener->onGeneric();
}

void EventBroadcaster::notifyLoadVolume(const VolumeData &volumeData) const
{
	for (VolumeLoadingListener *const pListener : __volumeLoadingListeners)
		pListener->onLoadVolume(volumeData);
}