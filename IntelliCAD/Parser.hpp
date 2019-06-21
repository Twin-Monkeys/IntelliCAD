/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: Parser.hpp
*	작성자			: 이세인
*	최종 수정일		: 19.03.22
*
*	여러가지 자료형의 상호 변환 함수를 제공
*/

#pragma once

#include <type_traits>
#include <atlstr.h>
#include <WinSock2.h>
#include "tstring.h"
#include "Color.hpp"

namespace Parser
{
	/// <summary>
	/// <para>tstring 타입의 문자열을 정수로 바꿀 수 있는지 여부를 반환한다.</para>
	/// </summary>
	/// <param name="str">정수로 바꿀 수 있는지 조사할 문자열</param>
	/// <returns>주어진 문자열을 정수로 바꿀 수 있는지 여부</returns>
	bool isAvailable_tstring$Int(const std::tstring &str);

	/// <summary>
	/// <para>CString 타입의 문자열을 정수로 바꿀 수 있는지 여부를 반환한다.</para>
	/// </summary>
	/// <param name="str">정수로 바꿀 수 있는지 조사할 문자열</param>
	/// <returns>주어진 문자열을 정수로 바꿀 수 있는지 여부</returns>
	bool isAvailable_CString$Int(const CString &str);

	/// <summary>
	/// <para>tstring 타입의 문자열을 정수로 변환한다.</para>
	/// <para>잘못된 형태의 문자열의 경우 값의 무결성을 보장하지 않는다.</para>
	/// <para>따라서 <c>Parser::isAvailable_tstring$Int()</c> 함수를 통해 값의 무결성을 먼저 판단한 후 이 함수를 사용하여야 한다.</para>
	/// <para>파입 파라미터의 경우 기본 타입은 int이며, 기타 호환 가능한 정수 타입으로 변경이 가능하다.</para>
	/// <para>호환되지 않는 타입을 입력하는 경우 본 함수는 컴파일되지 않는다.</para>
	/// </summary>
	/// <param name="str">정수로 바꿀 문자열</param>
	/// <returns>주어진 문자열을 정수로 변환한 값</returns>
	template <typename T = int>
	T tstring$int(const std::tstring &str);

	/// <summary>
	/// <para>CString 타입의 문자열을 정수로 변환한다.</para>
	/// <para>잘못된 형태의 문자열의 경우 값의 무결성을 보장하지 않는다.</para>
	/// <para>따라서 <c>Parser::isAvailable_tstring$Int()</c> 함수를 통해 값의 무결성을 먼저 판단한 후 이 함수를 사용하여야 한다.</para>
	/// <para>파입 파라미터의 경우 기본 타입은 int이며, 기타 호환 가능한 정수 타입으로 변경이 가능하다.</para>
	/// <para>호환되지 않는 타입을 입력하는 경우 본 함수는 컴파일되지 않는다.</para>
	/// </summary>
	/// <param name="str">정수로 바꿀 문자열</param>
	/// <returns>주어진 문자열을 정수로 변환한 값</returns>
	template <typename T = int>
	int CString$int(const CString &str);

	template <typename T>
	T CString$float(const CString& str);

	/// <summary>
	/// LPCSTR 타입의 문자열을 tstring 타입의 문자열로 변환한다.
	/// </summary>
	/// <param name="str">LPCSTR 타입 문자열</param>
	/// <returns>tstring 타입 문자열</returns>
	std::tstring LPCSTR$tstring(const LPCSTR str);

	std::tstring charString$tstring(const char *const str);

	/// <summary>
	/// CString 타입의 문자열을 tstring 타입의 문자열로 변환한다.
	/// </summary>
	/// <param name="str">CString 타입 문자열</param>
	/// <returns>tstring 타입 문자열</returns>
	std::tstring CString$tstring(const CString &str);

	/// <summary>
	/// LPCTSTR 타입의 문자열을 tstring 타입의 문자열로 변환한다.
	/// </summary>
	/// <param name="str">LPCTSTR 타입 문자열</param>
	/// <returns>tstring 타입 문자열</returns>
	std::tstring LPCTSTR$tstring(const LPCTSTR str);

	/// <summary>
	/// tstring 타입의 문자열을 string 타입의 문자열로 변환한다.
	/// </summary>
	/// <param name="str">tstring 타입 문자열</param>
	/// <returns>string 타입 문자열</returns>
	std::string tstring$string(const std::tstring &str);

	CString tstring$CString(const std::tstring &str);

	std::tstring string$tstring(const std::string &str);

	/// <summary>
	/// IN_ADDR 값으로 표현되어 있는 ip 정보를 tstring 타입의 문자열로 변환한다.
	/// </summary>
	/// <param name="sin_addr">IN_ADDR 값으로 표현되어 있는 ip</param>
	/// <returns>tstring 문자열로 표현된 ip</returns>
	std::tstring sin_addr$ipString(IN_ADDR sin_addr);

	/// <summary>
	/// USHORT 값으로 표현되어 있는 포트 정보를 tstring 타입의 문자열로 변환한다.
	/// </summary>
	/// <param name="sin_port">USHORT 값으로 표현되어 있는 포트</param>
	/// <returns>tstring 문자열로 표현된 포트</returns>
	std::tstring sin_port$portString(USHORT sin_port);

	/// <summary>
	/// tstring 타입의 문자열로 표현된 ip를 IN_ADDR 값으로 변환한다.
	/// </summary>
	/// <param name="ipString">tstring 타입의 문자열로 표현된 ip</param>
	/// <returns>IN_ADDR 값으로 표현되어 있는 ip</returns>
	IN_ADDR ipString$sin_addr(const std::tstring &ipString);

	/// <summary>
	/// tstring 타입의 문자열로 표현된 포트를 USHORT 값으로 변환한다.
	/// </summary>
	/// <param name="portString">tstring 타입의 문자열로 표현된 포트</param>
	/// <returns>USHORT 값으로 표현되어 있는 포트</returns>
	USHORT portString$sin_port(const std::tstring &portString);

	Color<float> COLORREF$Color(const COLORREF color);
	COLORREF Color$COLORREF(const Color<float> &color);

	template <typename T>
	T tstring$int(const std::tstring &str)
	{
		static_assert(std::is_integral_v<T>,
			"The type parameter T must be sort of integer");

		return static_cast<T>(_ttoi(str.c_str()));
	}

	template <typename T>
	T CString$int(const CString& str)
	{
		static_assert(std::is_integral_v<T>,
			"The type parameter T must be sort of integer");

		return static_cast<T>(_ttoi(str));
	}

	template <typename T>
	T CString$float(const CString& str) 
	{
		static_assert(std::is_floating_point_v<T>,
			"The type parameter T must be sort of float");

		return static_cast<T>(_ttof(str));
	}
};