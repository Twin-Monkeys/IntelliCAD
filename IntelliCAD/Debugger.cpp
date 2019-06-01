/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: Debugger.cpp
*	작성자			: 이세인
*	최종 수정일		: 19.04.06
*
*	디버깅용 헤더
*/

#include "stdafx.h"
#include "Debugger.h"

using namespace std;

namespace Debugger
{
	void popMessageBox(const tstring &message, const tstring &title)
	{
		MessageBox(nullptr, message.c_str(), title.c_str(), MB_ICONINFORMATION | MB_OK);
	}
}