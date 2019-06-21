#include "ClientDBManager.h"
#include "Constant.h"

using namespace std;

bool ClientDBManager::load(const tstring &rootDir, const bool createIfNotExistent)
{
	if (isLoaded())
		return true;

	_loaded = __configMgr.load(rootDir, createIfNotExistent);
	return _loaded;
}

bool ClientDBManager::load(const bool createIfNotExistent)
{
	return load(Constant::DB::ROOT_DIR, createIfNotExistent);
}

ConfigDBManager &ClientDBManager::getConfigManager()
{
	return __configMgr;
}

bool ClientDBManager::store()
{
	return __configMgr.store();
}

void ClientDBManager::close()
{
	__configMgr.close();
}