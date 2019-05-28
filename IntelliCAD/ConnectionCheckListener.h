/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: ConnectionCheckListener.h
*	작성자			: 이세인
*	최종 수정일		: 19.05.02
*
*	ConnectionCheckListener
*/

#pragma once

class ConnectionCheckListener
{
public:
	virtual void onConnectionCheck() = 0;
};