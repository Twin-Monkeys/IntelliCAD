#include "UserInfo.h"

using namespace std;

UserInfo::UserInfo(
	const tstring &id, const tstring &passwd,
	const tstring &name, const tstring &gender, const tstring &age) :
	id(id), passwd(passwd), name(name), gender(gender), age(age)
{}

UserInfo::UserInfo(ReadStream &stream)
{
	stream.getAs(__idLength);
	stream.getAs(__passwdLength);
	stream.getAs(__nameLength);
	stream.getAs(__genderLength);
	stream.getAs(__ageLength);

	id.resize(__idLength);
	stream.get(id.data(), __idLength * static_cast<int>(sizeof(int)));

	passwd.resize(__passwdLength);
	stream.get(passwd.data(), __passwdLength * static_cast<int>(sizeof(int)));

	name.resize(__nameLength);
	stream.get(name.data(), __nameLength * static_cast<int>(sizeof(int)));

	gender.resize(__genderLength);
	stream.get(gender.data(), __genderLength * static_cast<int>(sizeof(int)));

	age.resize(__ageLength);
	stream.get(age.data(), __ageLength * static_cast<int>(sizeof(int)));
}

vector<ElementMeta> UserInfo::_getStreamMeta() const
{
	__idLength = static_cast<int>(id.size());
	__passwdLength = static_cast<int>(passwd.size());
	__nameLength = static_cast<int>(name.size());
	__genderLength = static_cast<int>(gender.size());
	__ageLength = static_cast<int>(age.size());

	return
	{
		{ sizeof(__idLength), &__idLength },
		{ sizeof(__passwdLength), &__passwdLength },
		{ sizeof(__nameLength), &__nameLength },
		{ sizeof(__genderLength), &__genderLength },
		{ sizeof(__ageLength), &__ageLength },

		{ __idLength * static_cast<int>(sizeof(int)), id.data() },
		{ __passwdLength * static_cast<int>(sizeof(int)), passwd.data() },
		{ __nameLength * static_cast<int>(sizeof(int)), name.data() },
		{ __genderLength * static_cast<int>(sizeof(int)), gender.data() },
		{ __ageLength * static_cast<int>(sizeof(int)), age.data() },
	};
}