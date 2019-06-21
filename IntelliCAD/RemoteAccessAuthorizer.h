#pragma once

#include "Account.h"
#include "AuthorizingResult.h"
#include "ServerFileDBSectionType.h"
#include "RequestingServerDBResult.h"

class RemoteAccessAuthorizer
{
private:
	const std::tstring &__serverIP = _T("127.0.0.1");
	const std::tstring &__serverPort = _T("9000");

	bool __authorized = false;
	Account __account;

public:
	AuthorizingResult authorize(const std::tstring &id, const std::tstring &password);

	bool isAuthorized() const;
	const Account &getAccount() const;

	RequestingServerDBResult requestServerDBFile(const ServerFileDBSectionType sectionType, const std::tstring &name);
};
