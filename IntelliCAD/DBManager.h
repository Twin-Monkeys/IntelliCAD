#pragma once

#include "tstring.h"

class DBManager
{
protected:
	bool _loaded = false;

public:
	virtual bool load(const std::tstring &rootDir, const bool createIfNotExistent) = 0;
	bool isLoaded() const;

	virtual bool store() = 0;
	virtual void close() = 0;
};
