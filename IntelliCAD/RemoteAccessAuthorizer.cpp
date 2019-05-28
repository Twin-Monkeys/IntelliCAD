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

		// �ݵ�� ���� ���!
		this_thread::sleep_for(100ms);

		IF_F_RET_F(network.isConnected());
	}

	// ���Ĵ� ��Ʈ��ũ ������ �Ǿ� �ִٰ� ����

	// id�� password �׽�Ʈ
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