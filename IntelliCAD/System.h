#pragma once

#include "AsyncTaskManager.hpp"
#include "EventBroadcaster.h"
#include "RemoteAccessAuthorizer.h"
#include "RenderingEngine.hpp"
#include "ClientNetwork.h"
#include "ClientDBManager.h"

class System
{
private:
	friend class CIntelliCADApp;
	friend class CLogDialog;

	CLogDialog *__pLogDlg = nullptr;

	static System __instance;

	System() = default;

	void __setLogDlgReference(CLogDialog &dlg);

	void __init();
	void __release();

public:
	class SystemContents
	{
	private:
		friend System;

		AsyncTaskManager *__pTaskMgr = nullptr;
		EventBroadcaster *__pEventBroadcaster = nullptr;
		ClientDBManager *__pClientDBManager = nullptr;
		RemoteAccessAuthorizer *__pRemoteAccessAuthorizer = nullptr;
		RenderingEngine *__pRenderingEngine = nullptr;
		ClientNetwork *__pClientNetwork = nullptr;

		void __init();
		void __release();

	public:
		AsyncTaskManager &getTaskManager();
		EventBroadcaster &getEventBroadcaster();
		ClientDBManager &getClientDBManager();
		RemoteAccessAuthorizer &getRemoteAccessAuthorizer();
		RenderingEngine &getRenderingEngine();
		ClientNetwork &getClientNetwork();
	}
	systemContents;

	bool printLog(const std::tstring &message);

	static System& getInstance();
	static System::SystemContents& getSystemContents();
};
