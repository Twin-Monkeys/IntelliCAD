/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: ServerConnectingListener.h
*	작성자			: 이세인
*	최종 수정일		: 19.03.06
*
*	서버와 연결이 되면 연결 결과를 반환받는 리스너
*/

#pragma once

#include "Socket.h"

/// <summary>
/// <para>ServerConnected 이벤트에 대한 처리 능력을 요구하는 인터페이스이다.</para>
/// <para>이 인터페이스를 상속 및 구현한 뒤 EventBroadcaster에 등록하면 이벤트에 대한 콜백이 자동으로 이루어진다.</para>
/// <para>이벤트 처리는 항상 메인 스레드를 통해 이루어져야 한다.</para>
/// <para>단, 연산 부하가 큰 작업이 필요하다면 반드시 AsyncTaskManager를 통해 비동기 처리해주어야 한다.</para>
/// </summary>
class ServerConnectingListener
{
public:
	/// <summary>
	/// <para>서버와의 연결 시도 이후 그 결과를 콜백받는 함수이다.</para>
	/// <para>이 함수는 순수 가상 함수이므로 ServerConnectingListener 인터페이스를 구현하는 클래스는</para>
	/// <para>이 함수에 대한 처리 루틴을 반드시 구현하여야 한다.</para>
	/// <para>파라미터로 전달되는 패킷은 서버와 연결된 패킷 객체이다.</para>
	/// <para>서버 정보가 잘못 되었거나 서버가 열려있지 않은 경우 nullptr가 전달된다.</para>
	/// </summary>
	/// <param name="pSocket">서버와 연결된 후 획득한 소켓 객체</param>
	virtual void onServerConnectionResult(std::shared_ptr<Socket> pSocket) = 0;
};