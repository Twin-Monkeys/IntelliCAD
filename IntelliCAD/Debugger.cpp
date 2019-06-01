/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	���ϸ�			: Debugger.cpp
*	�ۼ���			: �̼���
*	���� ������		: 19.04.06
*
*	������ ���
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