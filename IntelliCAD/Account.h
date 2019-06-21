#pragma once

#include "tstring.h"
#include "Serializable.h"

class Account : public Serializable
{
private:
	mutable int __idLength;
	mutable int __passwordLength;
	mutable int __nameLength;

protected:
	virtual std::vector<ElementMeta> _getStreamMeta() const override;

public:
	std::tstring id;
	std::tstring password;
	std::tstring name;

	Account() = default;
	Account(const std::tstring &id, const std::tstring &passwd, const std::tstring &name);
	
	explicit Account(ReadStream &stream);
};