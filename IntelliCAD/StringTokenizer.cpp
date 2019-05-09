/*
*	Copyright (C) Sein Lee. All right reserved.
*
*	파일명			: StringTokenizer.cpp
*	작성자			: 이세인
*	최종 수정일		: 18.06.12
*
*	문자열 분리 유틸 클래스
*/

#include "stdafx.h"
#include "StringTokenizer.h"

using namespace std;

StringTokenizer::StringTokenizer(const tstring &initialString)
	: iss(initialString)
{}

StringTokenizer::StringTokenizer(const std::tstring &initialString, const TCHAR delimiter)
	: iss(initialString), delimiter(delimiter)
{}

void StringTokenizer::setString(const tstring &newString)
{
	iss.str(newString);
	iss.clear();
}

void StringTokenizer::setString(const TCHAR* const newString)
{
	iss.str(newString);
	iss.clear();
}

void StringTokenizer::setDelimiter(const TCHAR newDelimiter)
{
	delimiter = newDelimiter;
}

tstring StringTokenizer::getNext()
{
	tstring retVal;
	getline(iss, retVal, delimiter);

	return retVal;
}

bool StringTokenizer::hasNext() const
{
	return (!iss.eof());
}

vector<tstring> StringTokenizer::splitRemains()
{
	vector<tstring> retVal;
	tstring buffer;

	while (getline(iss, buffer, delimiter))
		retVal.emplace_back(buffer);

	return retVal;
}

StringTokenizer& StringTokenizer::operator=(const tstring &newString)
{
	setString(newString);

	return *this;
}