/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	���ϸ�			: EventBroadcaster.h
*	�ۼ���			: �̼���
*	���� ������		: 19.05.24
*
*	�̺�Ʈ ���Ŀ� Ŭ����
*/

#pragma once

#include <set>
#include <memory>
#include "SystemInitListener.h"
#include "GenericListener.h"
#include "VolumeLoadedListener.h"
#include "ServerConnectingListener.h"
#include "ConnectionCheckListener.h"
#include "ConnectionClosedListener.h"
#include "InitSlicingPointListener.h"
#include "InitSliceViewAnchorListener.h"
#include "RequestScreenUpdateListener.h"
#include "UpdateSliceTransferFunctionListener.h"
#include "InitSliceTransferFunctionListener.h"
#include "UpdateVolumeTransferFunctionListener.h"
#include "InitVolumeTransferFunctionListener.h"
#include "LoginSuccessListener.h"
#include "SystemDestroyListener.h"
#include "StreamTransmittingListener.h"
#include "StreamTransmissionFinishedListener.h"
#include "UpdateSlicingPointFromViewListener.h"
#include "UpdateAnchorFromViewListener.h"
#include "InitLightListener.h"
#include "UpdateLightListener.h"

/// <summary>
/// <para>���� �������� �߻��ϴ� �̺�Ʈ�� ��� ����ϴ� Ŭ�����̴�.</para>
/// <para>�ֿ� ����� �̺�Ʈ ������ ��� �� ����, �׸��� �̺�Ʈ �߻��̴�.</para>
/// </summary>
class EventBroadcaster
{
private:
	friend class System;

	/// <summary>
	/// ��ϵǾ� �ִ� SystemListener ��ü set
	/// </summary>
	std::set<SystemInitListener *> __systemInitListeners;
	std::set<SystemDestroyListener *> __systemDestroyListeners;

	/// <summary>
	/// ��ϵǾ� �ִ� GenericListener ��ü set
	/// </summary>
	std::set<GenericListener *> __genericListeners;

	std::set<VolumeLoadedListener *> __volumeLoadedListeners;

	/// <summary>
	/// ��ϵǾ� �ִ� ServerConnectingListener ��ü set
	/// </summary>
	std::set<ServerConnectingListener *> __serverConnectingListeners;

	/// <summary>
	/// ��ϵǾ� �ִ� ConnectionCheckListener ��ü set
	/// </summary>
	std::set<ConnectionCheckListener *> __connectionCheckListeners;

	std::set<ConnectionClosedListener *> __connectionClosedListeners;

	std::set<InitSlicingPointListener *> __initSlicingPointListeners;
	std::set<InitSliceViewAnchorListener *> __changeSliceViewAnchorListeners;

	std::set<RequestScreenUpdateListener *> __requestScreenUpdateListeners;

	std::set<UpdateSliceTransferFunctionListener *> __updateTransferFunctionListeners;
	std::set<InitSliceTransferFunctionListener *> __initTransferFunctionListeners;

	std::set<UpdateVolumeTransferFunctionListener *> __updateVolumeTransferFunctionListener;
	std::set<InitVolumeTransferFunctionListener *> __initVolumeTransferFunctionListener;

	std::set<LoginSuccessListener *> __loginSuccessListeners;

	std::set<StreamTransmittingListener *> __streamTransmittingListeners;
	std::set<StreamTransmissionFinishedListener *> __streamTransmissionFinishedListeners;

	std::set<UpdateSlicingPointFromViewListener *> __updateSlicingPointFromViewListeners;
	std::set<UpdateAnchorFromViewListener *> __updateAnchorFromViewListeners;

	std::set<InitLightListener *> __initLightListeners;
	std::set<UpdateLightListener *> __updateLightListeners;

	bool __addSystemInitListener(SystemInitListener &listener);
	bool __removeSystemInitListener(SystemInitListener &listener);
	void __notifySystemInit() const;
	void __notifySystemDestroy() const;

public:

	bool addSystemDestroyListener(SystemDestroyListener &listener);

	/// <summary>
	/// <para>GenericListener�� ���� ����Ѵ�. ��ü�� ���������� ��ϵ� ��� true�� ��ȯ�Ѵ�.</para>
	/// <para>���� ��ü�� �̹� ��ϵǾ� �־� �ְų�, ���� ������ ���� ���������� ��ϵ��� ���� ��� false�� ��ȯ�Ѵ�.</para>
	/// </summary>
	/// <param name="listener">����� GenericListener ��ü</param>
	/// <returns>��ü ��� ���� ����</returns>
	bool addGenericListener(GenericListener &listener);

	bool addVolumeLoadedListener(VolumeLoadedListener &listener);

	/// <summary>
	/// <para>ServerConnectingListener�� ���� ����Ѵ�. ��ü�� ���������� ��ϵ� ��� true�� ��ȯ�Ѵ�.</para>
	/// <para>���� ��ü�� �̹� ��ϵǾ� �־� �ְų�, ���� ������ ���� ���������� ��ϵ��� ���� ��� false�� ��ȯ�Ѵ�.</para>
	/// </summary>
	/// <param name="listener">����� ServerConnectingListener ��ü</param>
	/// <returns>��ü ��� ���� ����</returns>
	bool addServerConnectingListener(ServerConnectingListener &listener);

	bool addConnectionCheckListener(ConnectionCheckListener &listener);
	bool addConnectionClosedListener(ConnectionClosedListener &listener);

	bool addInitSlicingPointListener(InitSlicingPointListener &listener);
	bool addInitSliceViewAnchorListener(InitSliceViewAnchorListener &listener);

	bool addRequestScreenUpdateListener(RequestScreenUpdateListener &listener);

	bool addUpdateSliceTransferFunctionListener(UpdateSliceTransferFunctionListener &listener);
	bool addInitSliceTransferFunctionListener(InitSliceTransferFunctionListener &listener);

	bool addUpdateVolumeTransferFunctionListener(UpdateVolumeTransferFunctionListener &listener);
	bool addInitVolumeTransferFunctionListener(InitVolumeTransferFunctionListener &listener);

	bool addLoginSuccessListener(LoginSuccessListener &listener);

	bool addStreamTransmittingListener(StreamTransmittingListener &listener);
	bool addStreamTransmissionFinishedListener(StreamTransmissionFinishedListener &listener);

	bool addUpdateSlicingPointFromViewListener(UpdateSlicingPointFromViewListener &listener);
	bool addUpdateAnchorFromViewListener(UpdateAnchorFromViewListener &listener);

	bool addInitLightListener(InitLightListener &listener);
	bool addUpdateLightListener(UpdateLightListener &listener);

	bool removeSystemDestroyListener(SystemDestroyListener &listener);

	/// <summary>
	/// <para>��ϵǾ� �ִ� GenericListener�� ��� �����Ѵ�. ��ü�� ���������� ��� ������ ��� true�� ��ȯ�Ѵ�.</para>
	/// <para>���� ��ü�� ��ϵǾ� ���� �ʰų�, ���� ������ ���� ���������� �������� ���� ��� false�� ��ȯ�Ѵ�.</para>
	/// </summary>
	/// <param name="listener">��� ������ GenericListener ��ü</param>
	/// <returns>��ü ��� ���� ���� ����</returns>
	bool removeGenericListener(GenericListener &listener);

	bool removeVolumeLoadedListener(VolumeLoadedListener &listener);

	/// <summary>
	/// <para>��ϵǾ� �ִ� ServerConnectingListener�� ��� �����Ѵ�. ��ü�� ���������� ��� ������ ��� true�� ��ȯ�Ѵ�.</para>
	/// <para>���� ��ü�� ��ϵǾ� ���� �ʰų�, ���� ������ ���� ���������� �������� ���� ��� false�� ��ȯ�Ѵ�.</para>
	/// </summary>
	/// <param name="listener">��� ������ ServerConnectingListener ��ü</param>
	/// <returns>��ü ��� ���� ���� ����</returns>
	bool removeServerConnectingListener(ServerConnectingListener &listener);

	bool removeConnectionCheckListener(ConnectionCheckListener &listener);
	bool removeConnectionClosedListener(ConnectionClosedListener &listener);

	bool removeInitSlicingPointListener(InitSlicingPointListener &listener);
	bool removeInitSliceViewAnchorListener(InitSliceViewAnchorListener &listener);

	bool removeRequestScreenUpdateListener(RequestScreenUpdateListener &listener);

	bool removeUpdateSliceTransferFunctionListener(UpdateSliceTransferFunctionListener &listener);
	bool removeInitSliceTransferFunctionListener(InitSliceTransferFunctionListener &listener);

	bool removeUpdateVolumeTransferFunctionListener(UpdateVolumeTransferFunctionListener &listener);
	bool removeInitVolumeTransferFunctionListener(InitVolumeTransferFunctionListener &listener);

	bool removeLoginSuccessListener(LoginSuccessListener &listener);

	bool removeStreamTransmittingListener(StreamTransmittingListener &listener);
	bool removeStreamTransmissionFinishedListener(StreamTransmissionFinishedListener &listener);

	bool removeUpdateSlicingPointFromViewListener(UpdateSlicingPointFromViewListener &listener);
	bool removeUpdateAnchorFromViewListener(UpdateAnchorFromViewListener &listener);

	bool removeInitLightListener(InitLightListener &listener);
	bool removeUpdateLightListener(UpdateLightListener &listener);

	/// <summary>
	/// <para>���� ��ϵǾ� �ִ� ��� GenericListener ��ü���� generic �̺�Ʈ�� ��ε�ĳ���� �Ѵ�.</para>
	/// <para>generic �̺�Ʈ�� Ư���� ��ɰ� �����Ǿ� ���� �ʱ� ������ ����� �����ε� ����� �����ϴ�.</para>
	/// </summary>
	void notifyGeneric() const;

	void notifyVolumeLoaded(const VolumeMeta &volumeMeta) const;

	/// <summary>
	/// <para>���� ��ϵǾ� �ִ� ��� ServerConnectingListener ��ü���� ServerConnected �̺�Ʈ�� ��ε�ĳ���� �Ѵ�.</para>
	/// <para>pSocket�� �������� ���� ���� ���� ���� ��ü�̴�.</para>
	/// <para>��ü ȹ�濡 ���������� ����ϱ� ���� nullptr�� ������ ���� �ִ�.</para>
	/// </summary>
	/// <param name="pSocket">������ ����� �� ȹ���� ���� ��ü</param>
	void notifyServerConnected(std::shared_ptr<Socket> pSocket) const;

	void notifyConnectionCheck() const;
	void notifyConnectionClosed(std::shared_ptr<Socket> pSocket) const;

	void notifyInitSlicingPoint(const Point3D &slicingPoint) const;
	void notifyInitSliceViewAnchor(const SliceAxis axis) const;

	void notifyRequestScreenUpdate(const RenderingScreenType targetType) const;

	void notifySliceTransferFunctionUpdate() const;
	void notifySliceTransferFunctionInit() const;

	void notifyUpdateVolumeTransferFunction(const ColorChannelType colorType) const;
	void notifyInitVolumeTransferFunction(const ColorChannelType colorType) const;

	void notifyLoginSuccess(const Account &account) const;

	void notifyStreamTransmitting(
		std::shared_ptr<const class Socket> who,
		TransmittingDirectionType directionType, const int transmittedSize) const;

	void notifyStreamTransmissionFinished(
		std::shared_ptr<const class Socket> who, TransmittingDirectionType directionType) const;

	void notifyUpdateSlicingPointFromView() const;
	void notifyUpdateAnchorFromView(const SliceAxis axis) const;

	void notifyInitLight();
	void notifyUpdateLight();
};