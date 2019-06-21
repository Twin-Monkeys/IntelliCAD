#pragma once

#include "XMLBasedDBManager.h"
#include "ConfigDBSectionType.h"

class ConfigDBManager : public XMLBasedDBManager
{
protected:
	virtual const std::tstring &_getXMLSubPath() const override;

public:
	std::tstring getAttribute(const ConfigDBSectionType type, const std::tstring &attributeName) const;
};
