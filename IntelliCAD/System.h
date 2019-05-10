#pragma once

#include "AsyncTaskManager.hpp"
#include "EventBroadcaster.h"
#include "RenderingEngine.h"
#include "ImageProcessor.h"

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

		GPUVolume *__pVolume = nullptr;
		RenderingEngine *__pRenderingEngine = nullptr;
		ImageProcessor *__pImageProcessor = nullptr;

		void __init();
		void __release();

	public:
		AsyncTaskManager &getTaskManager();
		EventBroadcaster &getEventBroadcaster();

		GPUVolume *getGPUVolumePtr();
		RenderingEngine &getRenderingEngine();
		ImageProcessor &getImageProcessor();

		GPUVolume *getVolume();
		
		void setVolume(GPUVolume *const pVolume);
	}
	systemContents;

	static System& getInstance();
	static System::SystemContents& getSystemContents();
};
