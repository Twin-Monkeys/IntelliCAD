/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	���ϸ�			: NetworkUtility.cpp
*	�ۼ���			: �̼���
*	���� ������		: 19.03.22
*
*	��Ʈ��ũ�� ��ƿ��Ƽ �Լ� ����
*/

#include "NetworkUtility.h"
#include "StringTokenizer.h"
#include "NumberUtility.hpp"
#include "Parser.hpp"
#include "MacroTransaction.h"

using namespace std;

namespace NetworkUtility
{
	bool checkIPValidation(const tstring &ipString)
	{
		IF_T_RET_F(ipString.empty());

		StringTokenizer tokenizer(ipString, _T('.'));
		vector<tstring> tokens = tokenizer.splitRemains();

		IF_T_RET_F(tokens.size() != 4);

		for (const tstring &token : tokens)
		{
			IF_F_RET_F(Parser::isAvailable_tstring$Int(token));
			const int VALUE = Parser::tstring$int(token);

			IF_T_RET_F(NumberUtility::isOutOfBound(VALUE, 0, 256));
		}

		return true;
	}

	bool checkPortValidation(const tstring &portString)
	{
		IF_T_RET_F(portString.empty());
		IF_F_RET_F(Parser::isAvailable_tstring$Int(portString));

		const int VALUE = Parser::tstring$int(portString);

		return NumberUtility::isInOfBound(VALUE, 0, 65536);
	}
}
