#include "DatabaseManager.h"

using namespace std;

bool DatabaseManager::__copyAll(
	const tstring &sourceRootPath, const tstring &targetRootPath)
{

}

DatabaseManager::DatabaseManager(const tstring &rootPath, const bool creating)
{

}

bool DatabaseManager::load(const std::tstring &rootPath, const bool creating)
{
	return true;
}

bool DatabaseManager::isLoaded() const
{
	return true;
}

const std::tstring &DatabaseManager::getRootPath() const
{

}

const std::tstring &DatabaseManager::getSectionPath(const DBSectionType sectionType) const
{

}

bool DatabaseManager::move(const tstring &newRootPath)
{

}

bool DatabaseManager::copy(const tstring &targetRootPath)
{

}

const std::tstring DatabaseManager::getAttribute(const ConfigSectionType type, const tstring &attributeName)
{
	if (attributeName == _T("server_ip"))
		return _T("127.0.0.1");

	if (attributeName == _T("server_port"))
		return _T("8000");

	return _T("");
}

bool DatabaseManager::store() const
{

}

bool DatabaseManager::close()
{

}