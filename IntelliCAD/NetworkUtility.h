/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: NetworkUtility.h
*	작성자			: 이세인
*	최종 수정일		: 19.03.22
*
*	네트워크용 유틸리티 함수 모음
*/

#pragma once

#include "tstring.h"

/// <summary>
/// 네트워크와 관련한 유틸 함수를 모아놓은 네임스페이스
/// </summary>
namespace NetworkUtility
{
	/// <summary>
	/// <para>주어진 문자열이 IP 포맷을 만족하는지 검사한다.</para>
	/// <para>IP 포맷은 IPv4 규격을 따른다.</para>
	/// </summary>
	/// <param name="ipString">검사 대상</param>
	/// <returns>주어진 문자열이 IP 포맷을 만족하는지 여부</returns>
	bool checkIPValidation(const std::tstring &ipString);

	/// <summary>
	/// 주어진 문자열이 포트 포맷을 만족하는지 검사한다.
	/// </summary>
	/// <param name="ipString">검사 대상</param>
	/// <returns>주어진 문자열이 포트 포맷을 만족하는지 여부</returns>
	bool checkPortValidation(const std::tstring &portString);
};
