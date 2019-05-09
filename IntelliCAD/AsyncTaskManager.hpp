/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	���ϸ�			: AsyncTaskManager.hpp
*	�ۼ���			: �̼���
*	���� ������		: 19.04.06
*
*	�۾��� �񵿱� ó���� ���� �۾� ���� Ŭ����
*/

#pragma once

#include <vector>
#include <utility>
#include <any>
#include <thread>
#include <mutex>
#include "TaskType.h"

using FinishedTask = std::pair<TaskType, std::any>;

/// <summary>
/// �񵿱� �۾��� ���� �����ٸ��� �����ϴ� �۾� ������ Ŭ�����̴�.
/// </summary>
class AsyncTaskManager
{
private:

	/// <summary>
	/// ������ �Ϸ�� �۾��鿡 ���� ť �����̳��̴�.
	/// (�۾��� ����, �۾� ���) ���� ��ϵǾ� �ִ�.
	/// </summary>
	std::vector<FinishedTask> __finishedTasks;

	/// <summary>
	/// ��ȣ ������ ���� mutex ��ü
	/// </summary>
	mutable std::mutex __mutex;

public:

	/// <summary>
	/// <para>��� �Լ��� ���� �񵿱� �۾��� �����ϸ�, ���ο� �����忡�� �۾��� ��� �� ��� ����ȴ�.</para>
	/// <para>�� �Լ��� ��ȯ Ÿ���� void�� �Լ��� ȣ�⸸ ������ �� �ִ�.</para>
	/// <para>�Ϸ�� �۾��� �Ϸ� �۾� ť�� ���δ�.</para>
	/// </summary>
	/// <param name="taskType">�۾��� ����</param>
	/// <param name="worker">�۾��� ������ ��ü</param>
	/// <param name="work">�۾�(��� �Լ�)</param>
	/// <param name="args">�۾��� ���� ����(��� �Լ� ȣ�⿡ �ʿ��� ���� ���)</param>
	template <typename Worker, typename Work, typename... Args>
	void run_void(TaskType taskType, Worker &&worker, Work work, Args&&... args);

	/// <summary>
	/// <para>���� �Լ��� �񵿱� �۾��� �����ϸ�, ���ο� �����忡�� �۾��� ��� �� ��� ����ȴ�.</para>
	/// <para>�� �Լ��� ��ȯ Ÿ���� void�� �Լ��� ȣ�⸸ ������ �� �ִ�.</para>
	/// <para>�Ϸ�� �۾��� �Ϸ� �۾� ť�� ���δ�.</para>
	/// </summary>
	/// <param name="taskType">�۾��� ����</param>
	/// <param name="work">�۾�(��� �Լ�)</param>
	/// <param name="args">�۾��� ���� ����(��� �Լ� ȣ�⿡ �ʿ��� ���� ���)</param>
	template <typename Work, typename... Args>
	void run_void_global(TaskType taskType, Work work, Args&&... args);

	/// <summary>
	/// <para>��� �Լ��� ���� �񵿱� �۾��� �����ϸ�, ���ο� �����忡�� �۾��� ��� �� ��� ����ȴ�.</para>
	/// <para>�Ϸ�� �۾��� �Ϸ� �۾� ť�� ���δ�.</para>
	/// </summary>
	/// <param name="taskType">�۾��� ����</param>
	/// <param name="worker">�۾��� ������ ��ü</param>
	/// <param name="work">�۾�(��� �Լ�)</param>
	/// <param name="args">�۾��� ���� ����(��� �Լ� ȣ�⿡ �ʿ��� ���� ���)</param>
	template <typename Worker, typename Work, typename... Args>
	void run(TaskType taskType, Worker &&worker, Work work, Args&&... args);

	/// <summary>
	/// <para>���� �Լ��� �񵿱� �۾��� �����ϸ�, ���ο� �����忡�� �۾��� ��� �� ��� ����ȴ�.</para>
	/// <para>�Ϸ�� �۾��� �Ϸ� �۾� ť�� ���δ�.</para>
	/// </summary>
	/// <param name="taskType">�۾��� ����</param>
	/// <param name="work">�۾�(�Լ�)</param>
	/// <param name="args">�۾��� ���� ����(�Լ� ȣ�⿡ �ʿ��� ���� ���)</param>
	template <typename Work, typename... Args>
	void run_global(TaskType taskType, Work work, Args&&... args);

	/// <summary>
	/// �Ϸ�� �۾����� �ִ��� ���θ� ��ȯ�Ѵ�.
	/// </summary>
	/// <returns>�Ϸ�� �۾����� �ִ��� ����</returns>
	bool hasFinishedTask() const;

	/// <summary>
	/// <para>ť�� ���� �տ� �ִ� �Ϸ� �۾��� �ϳ� �����´�.</para>
	/// <para>�Ϸ� �۾��� ���� ������ (�۾��� ����, �۾� ���) ������ ǥ���ȴ�.</para>
	/// <para>�۾� ����� �񵿱� �۾��� ��û�� ����� ��� �Լ��� ��ȯ ���̴�.</para>
	/// </summary>
	/// <returns>ť�� ���� �տ� �ִ� �Ϸ� �۾�</returns>
	FinishedTask getNextFinishedTask();

	/// <summary>
	/// <para>���� ť�� �׿��ִ� ��� �Ϸ� �۾� ������ ��ȯ�Ѵ�.</para>
	/// </summary>
	/// <returns>���� ť�� �׿��ִ� ��� �Ϸ� �۾� ����</returns>
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

