#include "RemoteAccessAuthorizer.h"
#include "System.h"
#include "MacroTransaction.h"
#include "Parser.hpp"

using namespace std;

bool RemoteAccessAuthorizer::authorize(const tstring &id, const tstring &password)
{
	ClientNetwork &network = System::getSystemContents().getClientNetwork();

	if (!network.isConnected())
	{
		if (!network.isCreated())
			IF_F_RET_F(network.createClientSocket(__serverIP, __serverPort));

		network.connect();

		// 반드시 수정 요망!
		this_thread::sleep_for(100ms);

		IF_F_RET_F(network.isConnected());
	}

	// 이후는 네트워크 연결이 되어 있다고 가정

	// id와 password 테스트
	//network.sendMSG(Parser::tstring$string(id).c_str(), ProtocolType::CONNECTION_CHECK);
	//network.sendMSG(Parser::tstring$string(password).c_str(), ProtocolType::CONNECTION_CHECK);

	return true;
}

bool RemoteAccessAuthorizer::isAuthorized() const
{
	return __authorized;
}

const UserInfo &RemoteAccessAuthorizer::getUserInfo() const
{
	return __userInfo;
}