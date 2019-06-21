/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: System.cpp
*	작성자			: 이세인
*	최종 수정일		: 19.03.22
*
*	시스템 모듈
*/

#include "stdafx.h"
#include "System.h"
#include "CLogDialog.h"

using namespace std;

System System::__instance;

void System::__setLogDlgReference(CLogDialog &dlg)
{
	__pLogDlg = &dlg;
}

void System::__init()
{
	systemContents.__init();
}

void System::__release()
{
	systemContents.__pEventBroadcaster->__notifySystemDestroy();
	systemContents.__release();
}

void System::SystemContents::__init()
{
	__pTaskMgr = new AsyncTaskManager();
	__pEventBroadcaster = new EventBroadcaster();
	__pClientDBManager = new ClientDBManager();
	__pRemoteAccessAuthorizer = new RemoteAccessAuthorizer();
	__pRenderingEngine = &RenderingEngine::getInstance();
	__pClientNetwork = &ClientNetwork::getInstance();

	// SystemInitListeners
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

ClientDBManager &System::SystemContents::getClientDBManager()
{
	return *__pClientDBManager;
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

void System::SystemContents::__release()
{
	if (__pEventBroadcaster)
	{
		delete __pEventBroadcaster;
		__pEventBroadcaster = nullptr;
	}

	if (__pClientDBManager)
	{
		delete __pClientDBManager;
		__pClientDBManager = nullptr;
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

bool System::printLog(const tstring &message)
{
	if (__pLogDlg)
		__pLogDlg->printLog(message);

	return __pLogDlg;
}

System& System::getInstance()
{
	return __instance;
}

System::SystemContents& System::getSystemContents()
{
	return __instance.systemContents;
}