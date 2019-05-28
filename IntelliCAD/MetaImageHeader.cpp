#include <fstream>
#include <sstream>
#include "MetaImageHeader.h"
#include "Parser.hpp"

using namespace std;

MetaImageHeader::MetaImageHeader(const tstring &path)
{
	load(path);
}

bool MetaImageHeader::load(const tstring &path)
{
	ifstream fin(path);

	if (!fin)
	{
		__loaded = false;
		return false;
	}

	stringstream ss;
	ss << fin.rdbuf();

	istringstream iss;
	string lineBuffer, tokenBuffer;
	while (getline(ss, lineBuffer))
	{
		iss.clear();
		iss.str(lineBuffer);
		getline(iss, tokenBuffer, ' ');

		const tstring KEY = Parser::string$tstring(tokenBuffer);
		__metaMap.emplace(KEY, vector<tstring>());
		
		vector<tstring> &tokenList = __metaMap[KEY];

		// discard '=' character
		getline(iss, tokenBuffer, ' ');

		while (getline(iss, tokenBuffer, ' '))
		{
			const tstring token = Parser::string$tstring(tokenBuffer);
			tokenList.emplace_back(token);
		}
	}

	__loaded = true;
	return true;
}

bool MetaImageHeader::isLoaded() const
{
	return __loaded;
}

const tstring &MetaImageHeader::getValue(const tstring &name, const int idx) const
{
	return __metaMap.at(name)[idx];
}