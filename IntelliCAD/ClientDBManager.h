#pragma once

#include "DBManager.h"
#include "ConfigDBManager.h"

class ClientDBManager : public DBManager
{
private:
	ConfigDBManager __configMgr;

public:
	virtual bool load(const std::tstring &rootDir, const bool createIfNotExistent = true) override;
	bool load(const bool createIfNotExistent = true);

	ConfigDBManager &getConfigManager();

	virtual bool store() override;
	virtual void close() override;
};
