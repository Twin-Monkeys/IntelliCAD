/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: Parser.cpp
*	작성자			: 이세인
*	최종 수정일		: 19.03.22
*
*	여러가지 자료형의 상호 변환 함수를 제공
*/

#include "stdafx.h"
#include <ws2tcpip.h>
#include "Parser.hpp"
#include "MacroTransaction.h"

using namespace std;

namespace Parser
{
	bool isAvailable_tstring$Int(const tstring &str)
	{
		IF_T_RET_F(str.empty());

		for (const TCHAR &c : str)
			IF_F_RET_F(_istdigit(c));

		return true;
	}

	bool isAvailable_CString$Int(const CString &str)
	{
		IF_T_RET_F(str.IsEmpty());

		const int LENGTH = str.GetLength();
		const LPCTSTR szPtr = str.GetString();

		for (int i = 0; i < LENGTH; i++)
			IF_F_RET_F(_istdigit(szPtr[i]));

		return true;
	}

	tstring LPCSTR$tstring(const LPCSTR str)
	{
		return tstring(CA2CT(str));
	}

	tstring CString$tstring(const CString &str)
	{
		return str.GetString();
	}

	tstring LPCTSTR$tstring(const LPCTSTR str)
	{
		return str;
	}

	string tstring$string(const tstring &str)
	{
		return string(CT2CA(str.c_str()));
	}

	tstring string$tstring(const std::string &str)
	{
		return tstring(CA2CT(str.c_str()));
	}

	tstring sin_addr$ipString(const IN_ADDR sin_addr)
	{
		char buffer[32];
		inet_ntop(AF_INET, &sin_addr, buffer, sizeof(buffer));

		return LPCSTR$tstring(buffer);
	}

	tstring sin_port$portString(const USHORT sin_port)
	{
		char buffer[32];
		sprintf_s(buffer, sizeof(buffer), "%hu", ntohs(sin_port));

		return LPCSTR$tstring(buffer);
	}

	IN_ADDR ipString$sin_addr(const tstring &ipString)
	{
		IN_ADDR retVal;

		string ipMB = Parser::tstring$string(ipString);
		inet_pton(AF_INET, ipMB.c_str(), &(retVal.s_addr));

		return retVal;
	}

	USHORT portString$sin_port(const tstring &portString)
	{
		return htons(tstring$Int<u_short>(portString));
	}
};