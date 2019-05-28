/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: System.cpp
*	작성자			: 이세인
*	최종 수정일		: 19.03.22
*
*	시스템 모듈
*/

#include "System.h"

System System::__instance;

void System::__init()
{
	__instance.systemContents.__init();
}

void System::__release()
{
	__instance.systemContents.__release();
}

void System::SystemContents::__init()
{
	__pTaskMgr = new AsyncTaskManager();
	__pEventBroadcaster = new EventBroadcaster();
	__pRemoteAccessAuthorizer = new RemoteAccessAuthorizer();
	__pRenderingEngine = &RenderingEngine::getInstance();
	__pClientNetwork = &ClientNetwork::getInstance();

	// SystemInitListeners
	__pEventBroadcaster->__addSystemInitListener(*__pRenderingEngine);
	__pEventBroadcaster->__addSystemInitListener(*__pClientNetwork);
	__pEventBroadcaster->__notifySystemInit();
}

AsyncTaskManager &System::SystemContents::getTaskManager()
{
	return *__pTaskMgr;
}

EventBroadcaster &System::SystemContents::getEventBroadcaster()
{
	return *__pEventBroadcaster;
}

RemoteAccessAuthorizer &System::SystemContents::getRemoteAccessAuthorizer()
{
	return *__pRemoteAccessAuthorizer;
}

RenderingEngine &System::SystemContents::getRenderingEngine()
{
	return *__pRenderingEngine;
}

ClientNetwork &System::SystemContents::getClientNetwork()
{
	return *__pClientNetwork;
}

void System::SystemContents::loadVolume(const VolumeData &volumeData)
{
	__pEventBroadcaster->notifyLoadVolume(volumeData);
}

void System::SystemContents::__release()
{
	if (__pEventBroadcaster)
	{
		delete __pEventBroadcaster;
		__pEventBroadcaster = nullptr;
	}

	if (__pTaskMgr)
	{
		delete __pTaskMgr;
		__pTaskMgr = nullptr;
	}

	if (__pRemoteAccessAuthorizer)
	{
		delete __pRemoteAccessAuthorizer;
		__pRemoteAccessAuthorizer = nullptr;
	}
}

System& System::getInstance()
{
	return __instance;
}

System::SystemContents& System::getSystemContents()
{
	return __instance.systemContents;
}