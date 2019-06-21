#pragma once

#include "DBManager.h"
#include "tinyxml2.h"

class XMLBasedDBManager : public DBManager
{
protected:
	tinyxml2::XMLDocument __doc;

	virtual const std::tstring &_getXMLSubPath() const = 0;

public:
	virtual bool load(const std::tstring &rootDir, const bool createIfNotExistent) override;

	virtual bool store() override;
	virtual void close() override;
};
