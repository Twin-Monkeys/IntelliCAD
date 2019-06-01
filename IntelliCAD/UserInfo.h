#pragma once

#include "tstring.h"
#include "Serializable.h"

class UserInfo : public Serializable
{
private:
	mutable int __idLength;
	mutable int __passwdLength;
	mutable int __nameLength;
	mutable int __genderLength;
	mutable int __ageLength;

protected:
	virtual std::vector<ElementMeta> _getStreamMeta() const override;

public:
	std::tstring id;
	std::tstring passwd;
	std::tstring name;
	std::tstring gender;
	std::tstring age;

	UserInfo() = default;
	UserInfo(
		const std::tstring &id, const std::tstring &passwd,
		const std::tstring &name, const std::tstring &gender, const std::tstring &age);
	
	explicit UserInfo(ReadStream &stream);
};