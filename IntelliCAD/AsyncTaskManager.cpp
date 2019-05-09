/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: AsyncTaskManager.cpp
*	작성자			: 이세인
*	최종 수정일		: 19.03.07
*
*	작업의 비동기 처리를 위한 작업 관리 클래스
*/

#include "stdafx.h"
#include "AsyncTaskManager.hpp"

using namespace std;

bool AsyncTaskManager::hasFinishedTask() const
{
	unique_lock<mutex> lock(__mutex);

	bool retVal = !(__finishedTasks.empty());
	lock.unlock();

	return retVal;
}

FinishedTask AsyncTaskManager::getNextFinishedTask()
{
	FinishedTask retVal = __finishedTasks.front();

	unique_lock<mutex> mut(__mutex);
	__finishedTasks.erase(__finishedTasks.begin());
	mut.unlock();

	return retVal;
}

vector<FinishedTask> AsyncTaskManager::getFinishedTasks()
{
	vector<FinishedTask> retVal;

	unique_lock<mutex> mut(__mutex);
	retVal.swap(__finishedTasks);
	mut.unlock();

	return retVal;
}