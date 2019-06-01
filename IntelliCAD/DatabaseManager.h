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

	// 내부에서 config 디렉토리의 config.xml 파일을 읽은 뒤 attributeName 태그에 해당하는 값을 읽어 반환
	const std::tstring getAttribute(const ConfigSectionType type, const std::tstring &attributeName);

	bool store() const;
	bool close();
};
