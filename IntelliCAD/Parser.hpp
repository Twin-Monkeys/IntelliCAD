/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	���ϸ�			: Parser.hpp
*	�ۼ���			: �̼���
*	���� ������		: 19.03.22
*
*	�������� �ڷ����� ��ȣ ��ȯ �Լ��� ����
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
	/// <para>tstring Ÿ���� ���ڿ��� ������ �ٲ� �� �ִ��� ���θ� ��ȯ�Ѵ�.</para>
	/// </summary>
	/// <param name="str">������ �ٲ� �� �ִ��� ������ ���ڿ�</param>
	/// <returns>�־��� ���ڿ��� ������ �ٲ� �� �ִ��� ����</returns>
	bool isAvailable_tstring$Int(const std::tstring &str);

	/// <summary>
	/// <para>CString Ÿ���� ���ڿ��� ������ �ٲ� �� �ִ��� ���θ� ��ȯ�Ѵ�.</para>
	/// </summary>
	/// <param name="str">������ �ٲ� �� �ִ��� ������ ���ڿ�</param>
	/// <returns>�־��� ���ڿ��� ������ �ٲ� �� �ִ��� ����</returns>
	bool isAvailable_CString$Int(const CString &str);

	/// <summary>
	/// <para>tstring Ÿ���� ���ڿ��� ������ ��ȯ�Ѵ�.</para>
	/// <para>�߸��� ������ ���ڿ��� ��� ���� ���Ἲ�� �������� �ʴ´�.</para>
	/// <para>���� <c>Parser::isAvailable_tstring$Int()</c> �Լ��� ���� ���� ���Ἲ�� ���� �Ǵ��� �� �� �Լ��� ����Ͽ��� �Ѵ�.</para>
	/// <para>���� �Ķ������ ��� �⺻ Ÿ���� int�̸�, ��Ÿ ȣȯ ������ ���� Ÿ������ ������ �����ϴ�.</para>
	/// <para>ȣȯ���� �ʴ� Ÿ���� �Է��ϴ� ��� �� �Լ��� �����ϵ��� �ʴ´�.</para>
	/// </summary>
	/// <param name="str">������ �ٲ� ���ڿ�</param>
	/// <returns>�־��� ���ڿ��� ������ ��ȯ�� ��</returns>
	template <typename T = int>
	T tstring$int(const std::tstring &str);

	/// <summary>
	/// <para>CString Ÿ���� ���ڿ��� ������ ��ȯ�Ѵ�.</para>
	/// <para>�߸��� ������ ���ڿ��� ��� ���� ���Ἲ�� �������� �ʴ´�.</para>
	/// <para>���� <c>Parser::isAvailable_tstring$Int()</c> �Լ��� ���� ���� ���Ἲ�� ���� �Ǵ��� �� �� �Լ��� ����Ͽ��� �Ѵ�.</para>
	/// <para>���� �Ķ������ ��� �⺻ Ÿ���� int�̸�, ��Ÿ ȣȯ ������ ���� Ÿ������ ������ �����ϴ�.</para>
	/// <para>ȣȯ���� �ʴ� Ÿ���� �Է��ϴ� ��� �� �Լ��� �����ϵ��� �ʴ´�.</para>
	/// </summary>
	/// <param name="str">������ �ٲ� ���ڿ�</param>
	/// <returns>�־��� ���ڿ��� ������ ��ȯ�� ��</returns>
	template <typename T = int>
	int CString$int(const CString &str);

	template <typename T>
	T CString$float(const CString& str);

	/// <summary>
	/// LPCSTR Ÿ���� ���ڿ��� tstring Ÿ���� ���ڿ��� ��ȯ�Ѵ�.
	/// </summary>
	/// <param name="str">LPCSTR Ÿ�� ���ڿ�</param>
	/// <returns>tstring Ÿ�� ���ڿ�</returns>
	std::tstring LPCSTR$tstring(const LPCSTR str);

	std::tstring charString$tstring(const char *const str);

	/// <summary>
	/// CString Ÿ���� ���ڿ��� tstring Ÿ���� ���ڿ��� ��ȯ�Ѵ�.
	/// </summary>
	/// <param name="str">CString Ÿ�� ���ڿ�</param>
	/// <returns>tstring Ÿ�� ���ڿ�</returns>
	std::tstring CString$tstring(const CString &str);

	/// <summary>
	/// LPCTSTR Ÿ���� ���ڿ��� tstring Ÿ���� ���ڿ��� ��ȯ�Ѵ�.
	/// </summary>
	/// <param name="str">LPCTSTR Ÿ�� ���ڿ�</param>
	/// <returns>tstring Ÿ�� ���ڿ�</returns>
	std::tstring LPCTSTR$tstring(const LPCTSTR str);

	/// <summary>
	/// tstring Ÿ���� ���ڿ��� string Ÿ���� ���ڿ��� ��ȯ�Ѵ�.
	/// </summary>
	/// <param name="str">tstring Ÿ�� ���ڿ�</param>
	/// <returns>string Ÿ�� ���ڿ�</returns>
	std::string tstring$string(const std::tstring &str);

	CString tstring$CString(const std::tstring &str);

	std::tstring string$tstring(const std::string &str);

	/// <summary>
	/// IN_ADDR ������ ǥ���Ǿ� �ִ� ip ������ tstring Ÿ���� ���ڿ��� ��ȯ�Ѵ�.
	/// </summary>
	/// <param name="sin_addr">IN_ADDR ������ ǥ���Ǿ� �ִ� ip</param>
	/// <returns>tstring ���ڿ��� ǥ���� ip</returns>
	std::tstring sin_addr$ipString(IN_ADDR sin_addr);

	/// <summary>
	/// USHORT ������ ǥ���Ǿ� �ִ� ��Ʈ ������ tstring Ÿ���� ���ڿ��� ��ȯ�Ѵ�.
	/// </summary>
	/// <param name="sin_port">USHORT ������ ǥ���Ǿ� �ִ� ��Ʈ</param>
	/// <returns>tstring ���ڿ��� ǥ���� ��Ʈ</returns>
	std::tstring sin_port$portString(USHORT sin_port);

	/// <summary>
	/// tstring Ÿ���� ���ڿ��� ǥ���� ip�� IN_ADDR ������ ��ȯ�Ѵ�.
	/// </summary>
	/// <param name="ipString">tstring Ÿ���� ���ڿ��� ǥ���� ip</param>
	/// <returns>IN_ADDR ������ ǥ���Ǿ� �ִ� ip</returns>
	IN_ADDR ipString$sin_addr(const std::tstring &ipString);

	/// <summary>
	/// tstring Ÿ���� ���ڿ��� ǥ���� ��Ʈ�� USHORT ������ ��ȯ�Ѵ�.
	/// </summary>
	/// <param name="portString">tstring Ÿ���� ���ڿ��� ǥ���� ��Ʈ</param>
	/// <returns>USHORT ������ ǥ���Ǿ� �ִ� ��Ʈ</returns>
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