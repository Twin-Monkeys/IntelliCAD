#pragma once

#include "tinyxml2.h"
#include "tstring.h"
#include "Parser.hpp"
#include "DBSectionType.h"
#include "ConfigSectionType.h"

class DatabaseManager
{
private:
	std::tstring __rootPath;

	std::tstring __configPath = _T("config/");
	std::tstring __documentPath = _T("doc/");

	bool __copyAll(const std::tstring &sourceRootPath, const std::tstring &targetRootPath);

public:
	DatabaseManager() = default;
	DatabaseManager(const std::tstring &rootPath, const bool creating = true);

	bool load(const std::tstring &rootPath, const bool creating = true);
	bool isLoaded() const;

	const std::tstring &getRootPath() const;
	const std::tstring &getSectionPath(const DBSectionType sectionType) const;

	bool move(const std::tstring &newRootPath);
	bool copy(const std::tstring &targetRootPath);

	// ���ο��� config ���丮�� config.xml ������ ���� �� attributeName �±׿� �ش��ϴ� ���� �о� ��ȯ
	const std::tstring getAttribute(const ConfigSectionType type, const std::tstring &attributeName);

	bool store() const;
	bool close();
};
