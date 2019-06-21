#include "stdafx.h"
#include "Account.h"

using namespace std;

Account::Account(const tstring &id, const tstring &password, const tstring &name) :
	id(id), password(password), name(name)
{}

Account::Account(ReadStream &stream)
{
	stream.getAs(__idLength);
	stream.getAs(__passwordLength);
	stream.getAs(__nameLength);

	id.resize(__idLength);
	stream.get(id.data(), __idLength * static_cast<int>(sizeof(int)));

	password.resize(__passwordLength);
	stream.get(password.data(), __passwordLength * static_cast<int>(sizeof(int)));

	name.resize(__nameLength);
	stream.get(name.data(), __nameLength * static_cast<int>(sizeof(int)));
}

vector<ElementMeta> Account::_getStreamMeta() const
{
	__idLength = static_cast<int>(id.size());
	__passwordLength = static_cast<int>(password.size());
	__nameLength = static_cast<int>(name.size());

	return
	{
		{ sizeof(__idLength), &__idLength },
		{ sizeof(__passwordLength), &__passwordLength },
		{ sizeof(__nameLength), &__nameLength },

		{ __idLength * static_cast<int>(sizeof(int)), id.data() },
		{ __passwordLength * static_cast<int>(sizeof(int)), password.data() },
		{ __nameLength * static_cast<int>(sizeof(int)), name.data() }
	};
}