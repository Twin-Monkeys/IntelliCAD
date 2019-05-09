/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: Debugger.h
*	작성자			: 이세인
*	최종 수정일		: 19.04.06
*
*	디버깅용 헤더
*/

#pragma once

#include "tstring.h"

namespace Debugger
{
	/// <summary>
	/// 간단한 메세지 박스를 출력한다.
	/// </summary>
	void popMessageBox(const std::tstring &message, const std::tstring &title=_T("Notice"));
}
