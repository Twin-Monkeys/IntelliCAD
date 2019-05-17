/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: EventBroadcaster.h
*	작성자			: 이세인
*	최종 수정일		: 19.03.06
*
*	이벤트 전파용 클래스
*/

#pragma once

#include <set>
#include <memory>
#include "GenericListener.h"
#include "VolumeLoadingListener.h"

/// <summary>
/// <para>응용 전역에서 발생하는 이벤트의 제어를 담당하는 클래스이다.</para>
/// <para>주요 기능은 이벤트 리스너 등록 및 해제, 그리고 이벤트 발생이다.</para>
/// </summary>
class EventBroadcaster
{
private:
	friend class System;

	/// <summary>
	/// 등록되어 있는 GenericListener 객체 set
	/// </summary>
	std::set<GenericListener *> __genericListeners;

	std::set<VolumeLoadingListener *> __volumeLoadingListeners;

public:

	/// <summary>
	/// <para>GenericListener를 새로 등록한다. 객체가 정상적으로 등록된 경우 true를 반환한다.</para>
	/// <para>현재 객체가 이미 등록되어 있어 있거나, 여러 이유로 인해 정상적으로 등록되지 않은 경우 false를 반환한다.</para>
	/// </summary>
	/// <param name="listener">등록할 GenericListener 객체</param>
	/// <returns>객체 등록 성공 여부</returns>
	bool addGenericListener(GenericListener &listener);

	bool addVolumeLoadingListener(VolumeLoadingListener &listener);


	/// <summary>
	/// <para>등록되어 있는 GenericListener를 등록 해제한다. 객체가 정상적으로 등록 해제된 경우 true를 반환한다.</para>
	/// <para>현재 객체가 등록되어 있지 않거나, 여러 이유로 인해 정상적으로 해제되지 않은 경우 false를 반환한다.</para>
	/// </summary>
	/// <param name="listener">등록 해제할 GenericListener 객체</param>
	/// <returns>객체 등록 해제 성공 여부</returns>
	bool removeGenericListener(GenericListener &listener);

	bool removeVolumeLoadingListener(VolumeLoadingListener &listener);

	/// <summary>
	/// <para>현재 등록되어 있는 모든 GenericListener 객체에게 generic 이벤트를 브로드캐스팅 한다.</para>
	/// <para>generic 이벤트는 특별한 기능과 연관되어 있지 않기 때문에 디버깅 용으로도 사용이 가능하다.</para>
	/// </summary>
	void notifyGeneric() const;

	void notifyLoadVolume(const VolumeData &volumeData) const;
};