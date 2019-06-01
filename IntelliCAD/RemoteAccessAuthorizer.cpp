#include "RemoteAccessAuthorizer.h"
#include "System.h"
#include "MacroTransaction.h"
#include "Parser.hpp"
#include "Constant.h"
#include "Debugger.h"

using namespace std;

AuthorizingResult RemoteAccessAuthorizer::authorize(const tstring &id, const tstring &password)
{
	/*
		TO-DO: 네트워크를 이용하여 아이디 및 패스워드 정보 인증, 이후 해당 사용자의 정보를 읽어온다.
		
		1. 서버와의 연결 확인
		2. 연결이 되어있지 않을 시 연결
		3. 연결이 되지 않는 경우 AuthorizingResult::FAILED_NETWORK_ERROR 반환
		4. 서버와 연결이 되었다면 id 정보 전송
		5. 서버에게 id 존재 유무 정보를 전달받고, id가 존재하지 않는다면 AuthorizingResult::FAILED_INVALID_ID 반환
		6. id가 존재한다면 password 정보 전송
		7. 올바른 password를 전달하였다면 해당 사용자의 정보를 서버에게 요청한다. (객체 전송(UserInfo 클래스의 인스턴스))
		8. 위 작업을 모두 성공하였다면 AuthorizingResult::SUCCESS를 반환하고 함수를 종료한다.

		현재 서버 DB에 등록된 유저 정보 (임시)
		<User>
			<ID>user1</ID>
			<PW>1234</PW>
			<NAME>홍길동</NAME>
			<GENDER>male</GENDER>
			<AGE>20</AGE>
		</User>
		<User>
			<ID>user2</ID>
			<PW>1234</PW>
			<NAME>어우동</NAME>
			<GENDER>female</GENDER>
			<AGE>21</AGE>
		</User>

		[서버 프로젝트에서 해보아야 할 것]

		서버에서 이와 같은 클라이언트 요청에 응답하기 위해선 서버에서도 상응하는 기능을 구현해야 함
		System::getSystemContents().getDatabaseManager(); 코드를 통해 DatabaseManager에 접근 가능
		DatabaseManager의 isLoaded() 및 load() 함수를 통해 로딩 유무 및 로드 수행 가능
		findId(id) 함수를 통해 id가 존재하는지 조사 가능
		getUserInfo(id) 함수를 통해 UserInfo를 얻을 수 있음.
		아래의 코드에서 DatabaseManager 사용에 대한 힌트를 얻을 수 있음.
		단 클라이언트와 서버의 DatabaseManager 클래스는 서로 다른 인터페이스를
		제공하므로 제공되는 전체 기능은 함수명을 직접 살펴볼 것

		user1, user2 모두 테스트 해볼 것.
	*/

	DatabaseManager &dbManger = System::getSystemContents().getDatabaseManager();
	ClientNetwork &network = System::getSystemContents().getClientNetwork();

	// 로딩이 되어 있지 않으면 로드 시도. 로드 실패 시 false 반환
	if (!dbManger.isLoaded())
		if (dbManger.load(Constant::Database::DB_ROOT, false))
			return AuthorizingResult::FAILED_DB_ERROR;

	// DatabaseManager로부터 ip 및 port 정보 얻어옴.
	const tstring SERVER_IP =
		dbManger.getAttribute(ConfigSectionType::NETWORK, _T("server_ip"));

	const int SERVER_PORT =
		Parser::tstring$Int(dbManger.getAttribute(ConfigSectionType::NETWORK, _T("server_port")));

	// TO-DO: Network logics
	UserInfo userInfoFromNetworkServer =
	{
		_T("아이디"),
		_T("비밀번호"),
		_T("이름"),
		_T("성별"),
		_T("나이")
	};

	this->__userInfo = userInfoFromNetworkServer;

	// UserInfo를 넘겨받은 뒤 Debugger 모듈을 통해 확인
	Debugger::popMessageBox(__userInfo.id, _T("id value"));
	Debugger::popMessageBox(__userInfo.passwd, _T("passwd value"));
	Debugger::popMessageBox(__userInfo.name, _T("name value"));
	Debugger::popMessageBox(__userInfo.gender, _T("gender value"));
	Debugger::popMessageBox(__userInfo.age, _T("age value"));

	return AuthorizingResult::SUCCESS;
}

bool RemoteAccessAuthorizer::isAuthorized() const
{
	return __authorized;
}

const UserInfo &RemoteAccessAuthorizer::getUserInfo() const
{
	return __userInfo;
}