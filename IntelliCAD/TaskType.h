/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: TaskType.h
*	작성자			: 이세인
*	최종 수정일		: 19.03.23
*
*	작업의 종류를 정의한 열거형 클래스
*/

#pragma once

/// <summary>
/// AsyncTaskManager에서 작업의 종류를 표현하기 위한 열거형 클래스
/// </summary>
enum class TaskType
{
	/// <summary>
	/// 서버와 connect를 시도하는 작업
	/// </summary>
	SERVER_CONNECTED,

	/// <summary>
	/// <para>범용으로 사용하는 작업 타입이다.</para>
	/// <para>특별한 기능과 연관되어 있지 않으며, 디버깅 용으로 사용 가능하다.</para>
	/// </summary>
	GENERIC,

	/// <summary>
	/// 비동기 처리를 요청하되, 처리 결과를 보고받고 싶지 않을 때 사용한다.
	/// </summary>
	INGNORED,

	/// <summary>
	/// 연결종료 필요
	/// </summary>
	CONNECTION_CLOSED
};