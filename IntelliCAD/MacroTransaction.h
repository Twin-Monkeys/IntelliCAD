/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: MacroTransaction.h
*	작성자			: 이세인
*	최종 수정일		: 19.03.23
*
*	자주 쓰이는 함수 및 문장을 매크로 함수 형태로 축약해놓은 헤더
*/

#pragma once

/// <summary>
/// <para>함수의 반환 값이 bool인 경우(호환 가능한 모든 bool 타입)에만 사용이 가능한 매크로 함수이다.</para>
/// <para>인자로 들어온 값이 true이면 아무런 작업도 수행하지 않는다.</para>
/// <para>인자로 들어온 값이 false이면 즉시 false를 반환한다.</para>
/// </summary>
/// <param name="expr">테스트할 bool 타입 표현식</param>
/// <returns>expr이 true이면 없음. false인 경우에만 false 반환</returns>
#define IF_F_RET_F(expr) if (!(expr)) return false

/// <summary>
/// <para>함수의 반환 값이 bool인 경우(호환 가능한 모든 bool 타입)에만 사용이 가능한 매크로 함수이다.</para>
/// <para>인자로 들어온 값이 true이면 즉시 false를 반환한다.</para>
/// <para>인자로 들어온 값이 false이면 아무런 작업도 수행하지 않는다.</para>
/// </summary>
/// <param name="expr">테스트할 bool 타입 표현식</param>
/// <returns>expr이 true이면 false 반환. false이면 없음</returns>
#define IF_T_RET_F(expr) if ((expr)) return false

/// <summary>
/// <para>함수의 반환 값이 bool인 경우(호환 가능한 모든 bool 타입)에만 사용이 가능한 매크로 함수이다.</para>
/// <para>인자로 들어온 값이 true이면 아무런 작업도 수행하지 않는다.</para>
/// <para>인자로 들어온 값이 false이면 즉시 true를 반환한다.</para>
/// </summary>
/// <param name="expr">테스트할 bool 타입 표현식</param>
/// <returns>expr이 true이면 없음. false인 경우에만 true 반환</returns>
#define IF_F_RET_T(expr) if (!(expr)) return true

/// <summary>
/// <para>함수의 반환 값이 bool인 경우(호환 가능한 모든 bool 타입)에만 사용이 가능한 매크로 함수이다.</para>
/// <para>인자로 들어온 값이 true이면 즉시 true를 반환한다.</para>
/// <para>인자로 들어온 값이 false이면 아무런 작업도 수행하지 않는다.</para>
/// </summary>
/// <param name="expr">테스트할 bool 타입 표현식</param>
/// <returns>expr이 true이면 true 반환. false이면 없음</returns>
#define IF_T_RET_T(expr) if ((expr)) return true

/// <summary>
/// <para>함수의 반환 값이 bool인 경우(호환 가능한 모든 bool 타입)에만 사용이 가능한 매크로 함수이다.</para>
/// <para>인자로 들어온 값이 true이면 아무런 작업도 수행하지 않는다.</para>
/// <para>인자로 들어온 값이 false이면 즉시 반환한다.</para>
/// </summary>
/// <param name="expr">테스트할 bool 타입 표현식</param>
#define IF_F_RET(expr) if (!(expr)) return

/// <summary>
/// <para>함수의 반환 값이 bool인 경우(호환 가능한 모든 bool 타입)에만 사용이 가능한 매크로 함수이다.</para>
/// <para>인자로 들어온 값이 true이면 즉시 반환한다.</para>
/// <para>인자로 들어온 값이 false이면 아무런 작업도 수행하지 않는다.</para>
/// </summary>
/// <param name="expr">테스트할 bool 타입 표현식</param>
#define IF_T_RET(expr) if ((expr)) return