#pragma once

#include "Account.h"

class LoginSuccessListener
{
public:
	virtual void onLoginSuccess(const Account &account) = 0;
};