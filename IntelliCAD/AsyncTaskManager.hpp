/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: AsyncTaskManager.hpp
*	작성자			: 이세인
*	최종 수정일		: 19.04.06
*
*	작업의 비동기 처리를 위한 작업 관리 클래스
*/

#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include "FinishedTask.h"

/// <summary>
/// 비동기 작업에 대한 스케줄링을 관장하는 작업 관리자 클래스이다.
/// </summary>
class AsyncTaskManager
{
private:

	/// <summary>
	/// 수행이 완료된 작업들에 대한 큐 컨테이너이다.
	/// (작업의 종류, 작업 결과) 쌍이 기록되어 있다.
	/// </summary>
	std::vector<FinishedTask> __finishedTasks;

	/// <summary>
	/// 상호 배제를 위한 mutex 객체
	/// </summary>
	mutable std::mutex __mutex;

public:

	/// <summary>
	/// <para>멤버 함수에 대한 비동기 작업을 수행하며, 새로운 스레드에게 작업을 명령 후 즉시 종료된다.</para>
	/// <para>이 함수는 반환 타입이 void인 함수의 호출만 지시할 수 있다.</para>
	/// <para>완료된 작업은 완료 작업 큐에 쌓인다.</para>
	/// </summary>
	/// <param name="taskType">작업의 종류</param>
	/// <param name="worker">작업을 수행할 객체</param>
	/// <param name="work">작업(멤버 함수)</param>
	/// <param name="args">작업에 대한 인자(멤버 함수 호출에 필요한 인자 목록)</param>
	template <typename Worker, typename Work, typename... Args>
	void run_void(TaskType taskType, Worker &&worker, Work work, Args&&... args);

	/// <summary>
	/// <para>전역 함수의 비동기 작업을 수행하며, 새로운 스레드에게 작업을 명령 후 즉시 종료된다.</para>
	/// <para>이 함수는 반환 타입이 void인 함수의 호출만 지시할 수 있다.</para>
	/// <para>완료된 작업은 완료 작업 큐에 쌓인다.</para>
	/// </summary>
	/// <param name="taskType">작업의 종류</param>
	/// <param name="work">작업(멤버 함수)</param>
	/// <param name="args">작업에 대한 인자(멤버 함수 호출에 필요한 인자 목록)</param>
	template <typename Work, typename... Args>
	void run_void_global(TaskType taskType, Work work, Args&&... args);

	/// <summary>
	/// <para>멤버 함수에 대한 비동기 작업을 수행하며, 새로운 스레드에게 작업을 명령 후 즉시 종료된다.</para>
	/// <para>완료된 작업은 완료 작업 큐에 쌓인다.</para>
	/// </summary>
	/// <param name="taskType">작업의 종류</param>
	/// <param name="worker">작업을 수행할 객체</param>
	/// <param name="work">작업(멤버 함수)</param>
	/// <param name="args">작업에 대한 인자(멤버 함수 호출에 필요한 인자 목록)</param>
	template <typename Worker, typename Work, typename... Args>
	void run(TaskType taskType, Worker &&worker, Work work, Args&&... args);

	/// <summary>
	/// <para>전역 함수의 비동기 작업을 수행하며, 새로운 스레드에게 작업을 명령 후 즉시 종료된다.</para>
	/// <para>완료된 작업은 완료 작업 큐에 쌓인다.</para>
	/// </summary>
	/// <param name="taskType">작업의 종류</param>
	/// <param name="work">작업(함수)</param>
	/// <param name="args">작업에 대한 인자(함수 호출에 필요한 인자 목록)</param>
	template <typename Work, typename... Args>
	void run_global(TaskType taskType, Work work, Args&&... args);

	/// <summary>
	/// 완료된 작업들이 있는지 여부를 반환한다.
	/// </summary>
	/// <returns>완료된 작업들이 있는지 여부</returns>
	bool hasFinishedTask() const;

	/// <summary>
	/// <para>큐의 가장 앞에 있는 완료 작업을 하나 꺼내온다.</para>
	/// <para>완료 작업에 대한 정보는 (작업의 종류, 작업 결과) 쌍으로 표현된다.</para>
	/// <para>작업 결과는 비동기 작업을 요청시 사용한 멤버 함수의 반환 값이다.</para>
	/// </summary>
	/// <returns>큐의 가장 앞에 있는 완료 작업</returns>
	FinishedTask getNextFinishedTask();

	/// <summary>
	/// <para>현재 큐에 쌓여있는 모든 완료 작업 정보를 반환한다.</para>
	/// </summary>
	/// <returns>현재 큐에 쌓여있는 모든 완료 작업 정보</returns>
	std::vector<FinishedTask> getFinishedTasks();
};

template <typename Worker, typename Work, typename... Args>
void AsyncTaskManager::run_void(const TaskType taskType, Worker &&worker, const Work work, Args&&... args)
{
	using namespace std;

	thread([&, this, taskType, work]
	{
		(worker.*work)(args...);

		lock_guard<mutex> mut(__mutex);
		__finishedTasks.emplace_back(taskType, 0);
	}).detach();
}

template <typename Work, typename... Args>
void AsyncTaskManager::run_void_global(const TaskType taskType, const Work work, Args&&... args)
{
	using namespace std;

	thread([&, this, taskType, work]
	{
		work(args...);

		lock_guard<mutex> mut(__mutex);
		__finishedTasks.emplace_back(taskType, 0);
	}).detach();
}

template <typename Worker, typename Work, typename... Args>
void AsyncTaskManager::run(const TaskType taskType, Worker &&worker, const Work work, Args&&... args)
{
	using namespace std;

	thread([&, this, taskType, work]
	{
		any result = (worker.*work)(args...);

		lock_guard<mutex> mut(__mutex);
		__finishedTasks.emplace_back(taskType, result);
	}).detach();
}

template <typename Work, typename... Args>
void AsyncTaskManager::run_global(const TaskType taskType, const Work work, Args&&... args)
{
	using namespace std;

	thread([&, this, taskType, work]
	{
		any result = work(args...);

		lock_guard<mutex> mut(__mutex);
		__finishedTasks.emplace_back(taskType, result);
	}).detach();
}

