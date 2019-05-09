/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: GenericListener.h
*	작성자			: 이세인
*	최종 수정일		: 19.04.06
*
*	범용 목적 이벤트 리스너
*/

#pragma once

/// <summary>
/// 범용으로 사용하기 위한 generic 이벤트에 대한 리스너 인터페이스이다.
/// </summary>
class GenericListener
{
public:
	/// <summary>
	/// generic 이벤트를 처리하는 함수이다.
	/// </summary>
	virtual void onGeneric() = 0;
};