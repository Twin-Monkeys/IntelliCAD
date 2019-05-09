/*
*	Copyright (C) Sein Lee. All right reserved.
*
*	���ϸ�			: StringTokenizer.h
*	�ۼ���			: �̼���
*	���� ������		: 18.06.12
*
*	���ڿ� �и� ��ƿ Ŭ����
*/

#pragma once

#include <vector>
#include <sstream>
#include "tstring.h"

/// <summary>
/// <para>���ڿ� �и��� ���� ��ƿ��Ƽ Ŭ�����̴�.</para>
/// <para>�⺻ delimiter(������)�� ���鹮��(space, ' ')�̴�.</para>
/// </summary>
class StringTokenizer
{
private:
	std::basic_istringstream<TCHAR> iss;
	TCHAR delimiter = _T(' ');

public:
	/// <summary>
	/// �⺻ �������̴�. ���Ŀ� �и��� ���ڿ��� delimiter�� �Է��ؾ� �Ѵ�.
	/// </summary>
	StringTokenizer() = default;

	/// <summary>
	/// ��ü�� ������ �Բ� �и��� ��� ���ڿ��� �Է��Ѵ�.
	/// </summary>
	/// <param name="initialString">�и��� ��� ���ڿ�</param>
	StringTokenizer(const std::tstring &initialString);

	/// <summary>
	/// ��ü�� ������ �Բ� �и��� ��� ���ڿ��� delimiter�� �Է��Ѵ�.
	/// </summary>
	/// <param name="initialString">�и��� ��� ���ڿ�</param>
	/// <param name="delimiter">��ū ���� ����</param>
	StringTokenizer(const std::tstring &initialString, TCHAR delimiter);

	/// <summary>
	/// ������ ���� �۾��� �ʱ�ȭ�ϰ� �и��� ��� ���ڿ��� ���� �Է��Ѵ�.
	/// </summary>
	/// <param name="newString">���ο� �и� ��� ���ڿ�</param>
	void setString(const std::tstring &newString);

	/// <summary>
	/// ������ ���� �۾��� �ʱ�ȭ�ϰ� �и��� ��� ���ڿ��� ���� �Է��Ѵ�.
	/// </summary>
	/// <param name="newString">���ο� �и� ��� ���ڿ�</param>
	void setString(const TCHAR* newString);

	/// <summary>
	/// delimiter�� �����Ѵ�.
	/// </summary>
	/// <param name="newDelimiter">���� ����� delimiter</param>
	void setDelimiter(const TCHAR newDelimiter);

	/// <summary>
	/// ���� �����ִ� ���ڿ�, ���� delimiter�� �������� ���� ��ū�� �����Ѵ�.
	/// </summary>
	/// <returns>�и��� ��ū</returns>
	std::tstring getNext();

	/// <summary>
	/// �и� ������ ��ū�� �����ִ��� �����Ѵ�.
	/// </summary>
	/// <returns>�и� ������ ��ū�� �����ִ��� ����</returns>
	bool hasNext() const;

	/// <summary>
	/// ���� �����ִ� ���ڿ��� ���� delimiter�� �������� ��� �и��� �� �ѹ��� ��ȯ�Ѵ�.
	/// </summary>
	/// <returns>�и��� ��ū ����</returns>
	std::vector<std::tstring> splitRemains();

	/// <summary>
	/// ������ ���� �۾��� �ʱ�ȭ�ϰ� �и��� ��� ���ڿ��� ���� �Է��Ѵ�.
	/// </summary>
	/// <param name="newString">���ο� �и� ��� ���ڿ�</param>
	/// <returns>���� ��ü�� ���۷���</returns>
	StringTokenizer& operator=(const std::tstring &newString);
};
