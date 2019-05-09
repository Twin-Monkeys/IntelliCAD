/*
*	Copyright (C) Sein Lee. All right reserved.
*
*	파일명			: StringTokenizer.h
*	작성자			: 이세인
*	최종 수정일		: 18.06.12
*
*	문자열 분리 유틸 클래스
*/

#pragma once

#include <vector>
#include <sstream>
#include "tstring.h"

/// <summary>
/// <para>문자열 분리를 위한 유틸리티 클래스이다.</para>
/// <para>기본 delimiter(구분자)는 공백문자(space, ' ')이다.</para>
/// </summary>
class StringTokenizer
{
private:
	std::basic_istringstream<TCHAR> iss;
	TCHAR delimiter = _T(' ');

public:
	/// <summary>
	/// 기본 생성자이다. 추후에 분리할 문자열과 delimiter를 입력해야 한다.
	/// </summary>
	StringTokenizer() = default;

	/// <summary>
	/// 객체의 생성과 함께 분리할 대상 문자열을 입력한다.
	/// </summary>
	/// <param name="initialString">분리할 대상 문자열</param>
	StringTokenizer(const std::tstring &initialString);

	/// <summary>
	/// 객체의 생성과 함께 분리할 대상 문자열과 delimiter를 입력한다.
	/// </summary>
	/// <param name="initialString">분리할 대상 문자열</param>
	/// <param name="delimiter">토큰 구분 문자</param>
	StringTokenizer(const std::tstring &initialString, TCHAR delimiter);

	/// <summary>
	/// 이전의 분할 작업을 초기화하고 분리할 대상 문자열을 새로 입력한다.
	/// </summary>
	/// <param name="newString">새로운 분리 대상 문자열</param>
	void setString(const std::tstring &newString);

	/// <summary>
	/// 이전의 분할 작업을 초기화하고 분리할 대상 문자열을 새로 입력한다.
	/// </summary>
	/// <param name="newString">새로운 분리 대상 문자열</param>
	void setString(const TCHAR* newString);

	/// <summary>
	/// delimiter를 변경한다.
	/// </summary>
	/// <param name="newDelimiter">새로 등록할 delimiter</param>
	void setDelimiter(const TCHAR newDelimiter);

	/// <summary>
	/// 현재 남아있는 문자열, 현재 delimiter를 기준으로 다음 토큰을 추출한다.
	/// </summary>
	/// <returns>분리된 토큰</returns>
	std::tstring getNext();

	/// <summary>
	/// 분리 가능한 토큰이 남아있는지 조사한다.
	/// </summary>
	/// <returns>분리 가능한 토큰이 남아있는지 여부</returns>
	bool hasNext() const;

	/// <summary>
	/// 현재 남아있는 문자열을 현재 delimiter를 기준으로 모두 분리한 뒤 한번에 반환한다.
	/// </summary>
	/// <returns>분리된 토큰 벡터</returns>
	std::vector<std::tstring> splitRemains();

	/// <summary>
	/// 이전의 분할 작업을 초기화하고 분리할 대상 문자열을 새로 입력한다.
	/// </summary>
	/// <param name="newString">새로운 분리 대상 문자열</param>
	/// <returns>현재 객체의 레퍼런스</returns>
	StringTokenizer& operator=(const std::tstring &newString);
};
