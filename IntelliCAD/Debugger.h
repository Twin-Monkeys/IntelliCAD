/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	���ϸ�			: Debugger.h
*	�ۼ���			: �̼���
*	���� ������		: 19.04.06
*
*	������ ���
*/

#pragma once

#include "tstring.h"

namespace Debugger
{
	/// <summary>
	/// ������ �޼��� �ڽ��� ����Ѵ�.
	/// </summary>
	void popMessageBox(const std::tstring &message, const std::tstring &title=_T("Notice"));
}
