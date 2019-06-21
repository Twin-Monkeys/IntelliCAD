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

void EventBroadcaster::__notifySystemDestroy() const
{
	for (SystemDestroyListener *const pListener : __systemDestroyListeners)
		pListener->onSystemDestroy();
}

bool EventBroadcaster::addSystemDestroyListener(SystemDestroyListener &listener)
{
	return __systemDestroyListeners.emplace(&listener).second;
}

bool EventBroadcaster::addGenericListener(GenericListener &listener)
{
	return __genericListeners.emplace(&listener).second;
}

bool EventBroadcaster::addVolumeLoadedListener(VolumeLoadedListener &listener)
{
	return __volumeLoadedListeners.emplace(&listener).second;
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

bool EventBroadcaster::addInitSlicingPointListener(InitSlicingPointListener &listener)
{
	return __initSlicingPointListeners.emplace(&listener).second;
}

bool EventBroadcaster::addInitSliceViewAnchorListener(InitSliceViewAnchorListener &listener)
{
	return __changeSliceViewAnchorListeners.emplace(&listener).second;
}

bool EventBroadcaster::addRequestScreenUpdateListener(RequestScreenUpdateListener &listener)
{
	return __requestScreenUpdateListeners.emplace(&listener).second;
}

bool EventBroadcaster::addUpdateSliceTransferFunctionListener(UpdateSliceTransferFunctionListener &listener)
{
	return __updateTransferFunctionListeners.emplace(&listener).second;
}

bool EventBroadcaster::addInitSliceTransferFunctionListener(InitSliceTransferFunctionListener &listener)
{
	return __initTransferFunctionListeners.emplace(&listener).second;
}

bool EventBroadcaster::addUpdateVolumeTransferFunctionListener(UpdateVolumeTransferFunctionListener &listener)
{
	return __updateVolumeTransferFunctionListener.emplace(&listener).second;
}

bool EventBroadcaster::addInitVolumeTransferFunctionListener(InitVolumeTransferFunctionListener &listener)
{
	return __initVolumeTransferFunctionListener.emplace(&listener).second;
}

bool EventBroadcaster::addLoginSuccessListener(LoginSuccessListener &listener)
{
	return __loginSuccessListeners.emplace(&listener).second;
}

bool EventBroadcaster::addStreamTransmittingListener(StreamTransmittingListener &listener)
{
	return __streamTransmittingListeners.emplace(&listener).second;
}

bool EventBroadcaster::addStreamTransmissionFinishedListener(StreamTransmissionFinishedListener &listener)
{
	return __streamTransmissionFinishedListeners.emplace(&listener).second;
}

bool EventBroadcaster::addUpdateSlicingPointFromViewListener(UpdateSlicingPointFromViewListener &listener)
{
	return __updateSlicingPointFromViewListeners.emplace(&listener).second;
}

bool EventBroadcaster::addUpdateAnchorFromViewListener(UpdateAnchorFromViewListener &listener)
{
	return __updateAnchorFromViewListeners.emplace(&listener).second;
}

bool EventBroadcaster::addInitLightListener(InitLightListener &listener)
{
	return __initLightListeners.emplace(&listener).second;
}

bool EventBroadcaster::addUpdateLightListener(UpdateLightListener &listener)
{
	return __updateLightListeners.emplace(&listener).second;
}

bool EventBroadcaster::removeSystemDestroyListener(SystemDestroyListener &listener)
{
	return __systemDestroyListeners.erase(&listener);
}

bool EventBroadcaster::removeGenericListener(GenericListener &listener)
{
	return __genericListeners.erase(&listener);
}

bool EventBroadcaster::removeVolumeLoadedListener(VolumeLoadedListener &listener)
{
	return __volumeLoadedListeners.erase(&listener);
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

bool EventBroadcaster::removeInitSlicingPointListener(InitSlicingPointListener &listener)
{
	return __initSlicingPointListeners.erase(&listener);
}

bool EventBroadcaster::removeInitSliceViewAnchorListener(InitSliceViewAnchorListener &listener)
{
	return __changeSliceViewAnchorListeners.erase(&listener);
}

bool EventBroadcaster::removeRequestScreenUpdateListener(RequestScreenUpdateListener &listener)
{
	return __requestScreenUpdateListeners.erase(&listener);
}

bool EventBroadcaster::removeUpdateSliceTransferFunctionListener(UpdateSliceTransferFunctionListener &listener)
{
	return __updateTransferFunctionListeners.erase(&listener);
}

bool EventBroadcaster::removeInitSliceTransferFunctionListener(InitSliceTransferFunctionListener &listener)
{
	return __initTransferFunctionListeners.erase(&listener);
}

bool EventBroadcaster::removeUpdateVolumeTransferFunctionListener(UpdateVolumeTransferFunctionListener &listener)
{
	return __updateVolumeTransferFunctionListener.erase(&listener);
}

bool EventBroadcaster::removeInitVolumeTransferFunctionListener(InitVolumeTransferFunctionListener &listener)
{
	return __initVolumeTransferFunctionListener.erase(&listener);
}

bool EventBroadcaster::removeLoginSuccessListener(LoginSuccessListener &listener)
{
	return __loginSuccessListeners.erase(&listener);
}

bool EventBroadcaster::removeStreamTransmittingListener(StreamTransmittingListener &listener)
{
	return __streamTransmittingListeners.erase(&listener);
}

bool EventBroadcaster::removeStreamTransmissionFinishedListener(StreamTransmissionFinishedListener &listener)
{
	return __streamTransmissionFinishedListeners.erase(&listener);
}

bool EventBroadcaster::removeUpdateSlicingPointFromViewListener(UpdateSlicingPointFromViewListener &listener)
{
	return __updateSlicingPointFromViewListeners.erase(&listener);
}

bool EventBroadcaster::removeUpdateAnchorFromViewListener(UpdateAnchorFromViewListener &listener)
{
	return __updateAnchorFromViewListeners.erase(&listener);
}

bool EventBroadcaster::removeInitLightListener(InitLightListener &listener)
{
	return __initLightListeners.erase(&listener);
}

bool EventBroadcaster::removeUpdateLightListener(UpdateLightListener &listener)
{
	return __updateLightListeners.erase(&listener);
}

void EventBroadcaster::notifyGeneric() const
{
	for (GenericListener *const pListener : __genericListeners)
		pListener->onGeneric();
}

void EventBroadcaster::notifyVolumeLoaded(const VolumeMeta &volumeMeta) const
{
	for (VolumeLoadedListener *const pListener : __volumeLoadedListeners)
		pListener->onVolumeLoaded(volumeMeta);
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

void EventBroadcaster::notifyInitSlicingPoint(const Point3D &slicingPoint) const
{
	for (InitSlicingPointListener *const pListener : __initSlicingPointListeners)
		pListener->onInitSlicingPoint(slicingPoint);
}

void EventBroadcaster::notifyInitSliceViewAnchor(const SliceAxis axis) const
{
	for (InitSliceViewAnchorListener *const pListener : __changeSliceViewAnchorListeners)
		pListener->onInitSliceViewAnchor(axis);
}

void EventBroadcaster::notifyRequestScreenUpdate(const RenderingScreenType targetType) const
{
	for (RequestScreenUpdateListener *const pListener : __requestScreenUpdateListeners)
		pListener->onRequestScreenUpdate(targetType);
}

void EventBroadcaster::notifySliceTransferFunctionUpdate() const
{
	for (UpdateSliceTransferFunctionListener *const pListener : __updateTransferFunctionListeners)
		pListener->onUpdateSliceTransferFunction();
}

void EventBroadcaster::notifySliceTransferFunctionInit() const
{
	for (InitSliceTransferFunctionListener *const pListener : __initTransferFunctionListeners)
		pListener->onInitSliceTransferFunction();
}

void EventBroadcaster::notifyUpdateVolumeTransferFunction(const ColorChannelType colorType) const
{
	for (UpdateVolumeTransferFunctionListener *const pListener : __updateVolumeTransferFunctionListener)
		pListener->onUpdateVolumeTransferFunction(colorType);
}

void EventBroadcaster::notifyInitVolumeTransferFunction(const ColorChannelType colorType) const
{
	for (InitVolumeTransferFunctionListener *const pListener : __initVolumeTransferFunctionListener)
		pListener->onInitVolumeTransferFunction(colorType);
}

void EventBroadcaster::notifyLoginSuccess(const Account &account) const
{
	for (LoginSuccessListener *const pListener : __loginSuccessListeners)
		pListener->onLoginSuccess(account);
}

void EventBroadcaster::notifyStreamTransmitting(
	shared_ptr<const Socket> who, TransmittingDirectionType directionType, const int transmittedSize) const
{
	for (StreamTransmittingListener *const pListener : __streamTransmittingListeners)
		pListener->onStreamTransmitting(who, directionType, transmittedSize);
}

void EventBroadcaster::notifyStreamTransmissionFinished(
	shared_ptr<const Socket> who, TransmittingDirectionType directionType) const
{
	for (StreamTransmissionFinishedListener *const pListener : __streamTransmissionFinishedListeners)
		pListener->onStreamTransmissionFinished(who, directionType);
}

void EventBroadcaster::notifyUpdateSlicingPointFromView() const
{
	for (UpdateSlicingPointFromViewListener *const pListener : __updateSlicingPointFromViewListeners)
		pListener->onUpdateSlicingPointFromView();
}

void EventBroadcaster::notifyUpdateAnchorFromView(const SliceAxis axis) const
{
	for (UpdateAnchorFromViewListener *const pListener : __updateAnchorFromViewListeners)
		pListener->onUpdateAnchorFromView(axis);
}

void EventBroadcaster::notifyInitLight()
{
	for (InitLightListener *const pListener : __initLightListeners)
		pListener->onInitLight();
}

void EventBroadcaster::notifyUpdateLight()
{
	for (UpdateLightListener *const pListener : __updateLightListeners)
		pListener->onUpdateLight();
}