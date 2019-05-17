#pragma once

#include "AsyncTaskManager.hpp"
#include "EventBroadcaster.h"
#include "RenderingEngine.h"

class System
{
private:
	friend class CIntelliCADApp;

	static System __instance;

	System() = default;

	static void __init();
	static void __release();

public:
	class SystemContents
	{
	private:
		friend System;

		AsyncTaskManager *__pTaskMgr = nullptr;
		EventBroadcaster *__pEventBroadcaster = nullptr;
		RenderingEngine *__pRenderingEngine = nullptr;

		void __init();
		void __release();

	public:
		AsyncTaskManager &getTaskManager();
		EventBroadcaster &getEventBroadcaster();
		RenderingEngine &getRenderingEngine();
		
		void loadVolume(const VolumeData &volumeData);
	}
	systemContents;

	static System& getInstance();
	static System::SystemContents& getSystemContents();
};
