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
		TO-DO: 클라이언트에서 서버로 서버의 DB에 있는 파일을 요청한다.
		
		로직

		3. ClientNetwork를 통해 ServerNetwork에게 sectionType과 name 전송.

		4. 서버와 연결이 불가능하면 RequestingServerDBResult::CONNECTION_ERROR 반환

		5. 서버에서 DBManager의 getPath() 함수를 사용하여 해당 파일의 상대 경로를 얻어옴

		6. 서버의 DBManager로부터 상대 경로를 얻을 수 없는 경우 RequestingServerDBResult::NON_EXISTENT 반환

		7. 본 함수(메인스레드) RequestingServerDBResult::SUCCESS 반환하고 함수 종료

		8. 상대 경로를 이용하여 서버에서 클라이언트로 파일 스트림 전송 (비동기 처리)

		9. 스트림 전송 도중 버퍼 사이즈만큼 전달받을 때 마다 (1KB였나?)
			이벤트 notify (새로운 이벤트 타입인 StreamTransmittingListener 이용)
			이것은 UI에서 파일 전송에 대한 진행 상황을 알 수 있도록 해주기 위함이다.

		10. 전송이 다 됬으면 전송이 완료됬음을 알리는 이벤트 notify
			(이것도 새로운 이벤트 타입 StreamTransmissionFinishedListener 이용)
	*/
	System::getInstance().printLog(_T("requestServerDBFile"));

	//  1. 현재 시스템이 인증되어 있는지 (isAuthorized() 함수 활용) 확인
	//	2. 인증되지 않은 경우 RequestingServerDBResult::NOT_AUTHORIZED 반환
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