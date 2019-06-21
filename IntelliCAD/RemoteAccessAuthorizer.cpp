#include "RemoteAccessAuthorizer.h"
#include "System.h"
#include "MacroTransaction.h"
#include "Parser.hpp"
#include "Constant.h"
#include "Debugger.h"

using namespace std;

AuthorizingResult RemoteAccessAuthorizer::authorize(const tstring &id, const tstring &password)
{
	ClientDBManager &dbManger = System::getSystemContents().getClientDBManager();
	ClientNetwork &network = System::getSystemContents().getClientNetwork();

	if (!dbManger.isLoaded())
		if (!dbManger.load())
			return AuthorizingResult::FAILED_DB_ERROR;

	const ConfigDBManager &configMgr = dbManger.getConfigManager();

	const tstring SERVER_IP =
		configMgr.getAttribute(ConfigDBSectionType::CLIENT_NETWORK, _T("server_ip"));

	const tstring SERVER_PORT =
		configMgr.getAttribute(ConfigDBSectionType::CLIENT_NETWORK, _T("server_port"));
	
	if (!network.isConnected())
	{
		if(!network.createClientSocket(SERVER_IP, SERVER_PORT))
			return AuthorizingResult::FAILED_NETWORK_ERROR;
		
		if (!network.connectBlocking())
			return AuthorizingResult::FAILED_NETWORK_ERROR;
	}

	const AuthorizingResult RESULT = network.loginRequest(__account, id, password);

	if (RESULT == AuthorizingResult::SUCCESS)
	{
		__authorized = true;
		System::getSystemContents().getEventBroadcaster().notifyLoginSuccess(__account);
	}
	
	return RESULT;
}

bool RemoteAccessAuthorizer::isAuthorized() const
{
	return __authorized;
}

const Account &RemoteAccessAuthorizer::getAccount() const
{
	return __account;
}

RequestingServerDBResult RemoteAccessAuthorizer::requestServerDBFile(
	const ServerFileDBSectionType sectionType, const tstring &name)
{
	/*
		TO-DO: Ŭ���̾�Ʈ���� ������ ������ DB�� �ִ� ������ ��û�Ѵ�.
		
		����

		3. ClientNetwork�� ���� ServerNetwork���� sectionType�� name ����.

		4. ������ ������ �Ұ����ϸ� RequestingServerDBResult::CONNECTION_ERROR ��ȯ

		5. �������� DBManager�� getPath() �Լ��� ����Ͽ� �ش� ������ ��� ��θ� ����

		6. ������ DBManager�κ��� ��� ��θ� ���� �� ���� ��� RequestingServerDBResult::NON_EXISTENT ��ȯ

		7. �� �Լ�(���ν�����) RequestingServerDBResult::SUCCESS ��ȯ�ϰ� �Լ� ����

		8. ��� ��θ� �̿��Ͽ� �������� Ŭ���̾�Ʈ�� ���� ��Ʈ�� ���� (�񵿱� ó��)

		9. ��Ʈ�� ���� ���� ���� �����ŭ ���޹��� �� ���� (1KB����?)
			�̺�Ʈ notify (���ο� �̺�Ʈ Ÿ���� StreamTransmittingListener �̿�)
			�̰��� UI���� ���� ���ۿ� ���� ���� ��Ȳ�� �� �� �ֵ��� ���ֱ� �����̴�.

		10. ������ �� ������ ������ �Ϸ������ �˸��� �̺�Ʈ notify
			(�̰͵� ���ο� �̺�Ʈ Ÿ�� StreamTransmissionFinishedListener �̿�)
	*/
	System::getInstance().printLog(_T("requestServerDBFile"));

	//  1. ���� �ý����� �����Ǿ� �ִ��� (isAuthorized() �Լ� Ȱ��) Ȯ��
	//	2. �������� ���� ��� RequestingServerDBResult::NOT_AUTHORIZED ��ȯ
	if (!isAuthorized()) {
		System::getInstance().printLog(_T("NOT_AUTHORIZED"));
		return RequestingServerDBResult::NOT_AUTHORIZED;
	}
	ClientNetwork &network = System::getSystemContents().getClientNetwork();

	if (!network.isConnected()) {
		System::getInstance().printLog(_T("CONNECTION_ERROR"));
		return RequestingServerDBResult::CONNECTION_ERROR;
	}

	if (network.requestServerDBFile(sectionType, name)) {
		return RequestingServerDBResult::SUCCESS;
	}
	else
		return RequestingServerDBResult::NON_EXISTENT;
}