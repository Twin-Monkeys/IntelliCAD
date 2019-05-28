#pragma once

#include "UserInfo.h"

class RemoteAccessAuthorizer
{
private:
	const std::tstring &__serverIP = _T("127.0.0.1");
	const std::tstring &__serverPort = _T("9000");

	bool __authorized = false;
	UserInfo __userInfo;

public:
	bool authorize(const std::tstring &id, const std::tstring &password);
	bool isAuthorized() const;
	const UserInfo &getUserInfo() const;
};
