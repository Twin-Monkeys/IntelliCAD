/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	���ϸ�			: EventBroadcaster.h
*	�ۼ���			: �̼���
*	���� ������		: 19.03.06
*
*	�̺�Ʈ ���Ŀ� Ŭ����
*/

#pragma once

#include <set>
#include <memory>
#include "GenericListener.h"
#include "SetVolumeListener.h"

/// <summary>
/// <para>���� �������� �߻��ϴ� �̺�Ʈ�� ��� ����ϴ� Ŭ�����̴�.</para>
/// <para>�ֿ� ����� �̺�Ʈ ������ ��� �� ����, �׸��� �̺�Ʈ �߻��̴�.</para>
/// </summary>
class EventBroadcaster
{
private:
	friend class System;

	/// <summary>
	/// ��ϵǾ� �ִ� GenericListener ��ü set
	/// </summary>
	std::set<GenericListener *> __genericListeners;

	std::set<SetVolumeListener *> __setVolumeListeners;

public:

	/// <summary>
	/// <para>GenericListener�� ���� ����Ѵ�. ��ü�� ���������� ��ϵ� ��� true�� ��ȯ�Ѵ�.</para>
	/// <para>���� ��ü�� �̹� ��ϵǾ� �־� �ְų�, ���� ������ ���� ���������� ��ϵ��� ���� ��� false�� ��ȯ�Ѵ�.</para>
	/// </summary>
	/// <param name="listener">����� GenericListener ��ü</param>
	/// <returns>��ü ��� ���� ����</returns>
	bool addGenericListener(GenericListener &listener);

	bool addSetVolumeListener(SetVolumeListener &listener);


	/// <summary>
	/// <para>��ϵǾ� �ִ� GenericListener�� ��� �����Ѵ�. ��ü�� ���������� ��� ������ ��� true�� ��ȯ�Ѵ�.</para>
	/// <para>���� ��ü�� ��ϵǾ� ���� �ʰų�, ���� ������ ���� ���������� �������� ���� ��� false�� ��ȯ�Ѵ�.</para>
	/// </summary>
	/// <param name="listener">��� ������ GenericListener ��ü</param>
	/// <returns>��ü ��� ���� ���� ����</returns>
	bool removeGenericListener(GenericListener &listener);

	bool removeSetVolumeListener(SetVolumeListener &listener);

	/// <summary>
	/// <para>���� ��ϵǾ� �ִ� ��� GenericListener ��ü���� generic �̺�Ʈ�� ��ε�ĳ���� �Ѵ�.</para>
	/// <para>generic �̺�Ʈ�� Ư���� ��ɰ� �����Ǿ� ���� �ʱ� ������ ����� �����ε� ����� �����ϴ�.</para>
	/// </summary>
	void notifyGeneric() const;

	void notifySetVolume(const GPUVolume *const pVolume) const;
};