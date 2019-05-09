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
	__pRenderingEngine = new RenderingEngine();
	__pImageProcessor = new ImageProcessor();
}

AsyncTaskManager &System::SystemContents::getTaskManager()
{
	return *__pTaskMgr;
}

EventBroadcaster &System::SystemContents::getEventBroadcaster()
{
	return *__pEventBroadcaster;
}

GPUVolume *System::SystemContents::getGPUVolumePtr()
{
	return __pVolume;
}

RenderingEngine &System::SystemContents::getRenderingEngine()
{
	return *__pRenderingEngine;
}

ImageProcessor &System::SystemContents::getImageProcessor()
{
	return *__pImageProcessor;
}

GPUVolume *System::SystemContents::getVolume()
{
	return __pVolume;
}

void System::SystemContents::setVolume(GPUVolume *const pVolume)
{
	if (__pVolume)
		delete __pVolume;

	__pVolume = pVolume;

	__pEventBroadcaster->notifySetVolume(pVolume);
}

void System::SystemContents::__release()
{
	if (__pRenderingEngine)
	{
		delete __pRenderingEngine;
		__pRenderingEngine = nullptr;
	}

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
}

System& System::getInstance()
{
	return __instance;
}

System::SystemContents& System::getSystemContents()
{
	return __instance.systemContents;
}