/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: EventBroadcaster.h
*	작성자			: 이세인
*	최종 수정일		: 19.05.24
*
*	이벤트 전파용 클래스
*/

#include "EventBroadcaster.h"
#include "MacroTransaction.h"

using namespace std;

bool EventBroadcaster::__addSystemInitListener(SystemInitListener &listener)
{
	return __systemInitListeners.emplace(&listener).second;

}

bool EventBroadcaster::__removeSystemInitListener(SystemInitListener &listener)
{
	return __systemInitListeners.erase(&listener);

}

void EventBroadcaster::__notifySystemInit() const
{
	for (SystemInitListener *const pListener : __systemInitListeners)
		pListener->onSystemInit();
}

bool EventBroadcaster::addGenericListener(GenericListener &listener)
{
	return __genericListeners.emplace(&listener).second;
}

bool EventBroadcaster::addVolumeLoadingListener(VolumeLoadingListener &listener)
{
	return __volumeLoadingListeners.emplace(&listener).second;
}

bool EventBroadcaster::addServerConnectingListener(ServerConnectingListener &listener)
{
	return __serverConnectingListeners.emplace(&listener).second;
}

bool EventBroadcaster::addConnectionCheckListener(ConnectionCheckListener &listener)
{
	return __connectionCheckListeners.emplace(&listener).second;
}

bool EventBroadcaster::addConnectionClosedListener(ConnectionClosedListener &listener)
{
	return __connectionClosedListeners.emplace(&listener).second;
}

bool EventBroadcaster::removeGenericListener(GenericListener &listener)
{
	return __genericListeners.erase(&listener);
}

bool EventBroadcaster::removeVolumeLoadingListener(VolumeLoadingListener &listener)
{
	return __volumeLoadingListeners.erase(&listener);
}

bool EventBroadcaster::removeServerConnectingListener(ServerConnectingListener &listener)
{
	return __serverConnectingListeners.erase(&listener);
}

bool EventBroadcaster::removeConnectionCheckListener(ConnectionCheckListener &listener)
{
	return __connectionCheckListeners.erase(&listener);
}

bool EventBroadcaster::removeConnectionClosedListener(ConnectionClosedListener &listener)
{
	return __connectionClosedListeners.erase(&listener);
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

void EventBroadcaster::notifyServerConnected(shared_ptr<Socket> pSocket) const
{
	for (ServerConnectingListener *const pListener : __serverConnectingListeners)
		pListener->onServerConnectionResult(pSocket);
}

void EventBroadcaster::notifyConnectionCheck() const
{
	for (ConnectionCheckListener *const pListener : __connectionCheckListeners)
		pListener->onConnectionCheck();
}

void EventBroadcaster::notifyConnectionClosed(shared_ptr<Socket> pSocket) const
{
	for (ConnectionClosedListener *const pListener : __connectionClosedListeners)
		pListener->onConnectionClosed(pSocket);
}