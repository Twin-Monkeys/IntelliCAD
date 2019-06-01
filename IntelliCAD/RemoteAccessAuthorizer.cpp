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
		TO-DO: ��Ʈ��ũ�� �̿��Ͽ� ���̵� �� �н����� ���� ����, ���� �ش� ������� ������ �о�´�.
		
		1. �������� ���� Ȯ��
		2. ������ �Ǿ����� ���� �� ����
		3. ������ ���� �ʴ� ��� AuthorizingResult::FAILED_NETWORK_ERROR ��ȯ
		4. ������ ������ �Ǿ��ٸ� id ���� ����
		5. �������� id ���� ���� ������ ���޹ް�, id�� �������� �ʴ´ٸ� AuthorizingResult::FAILED_INVALID_ID ��ȯ
		6. id�� �����Ѵٸ� password ���� ����
		7. �ùٸ� password�� �����Ͽ��ٸ� �ش� ������� ������ �������� ��û�Ѵ�. (��ü ����(UserInfo Ŭ������ �ν��Ͻ�))
		8. �� �۾��� ��� �����Ͽ��ٸ� AuthorizingResult::SUCCESS�� ��ȯ�ϰ� �Լ��� �����Ѵ�.

		���� ���� DB�� ��ϵ� ���� ���� (�ӽ�)
		<User>
			<ID>user1</ID>
			<PW>1234</PW>
			<NAME>ȫ�浿</NAME>
			<GENDER>male</GENDER>
			<AGE>20</AGE>
		</User>
		<User>
			<ID>user2</ID>
			<PW>1234</PW>
			<NAME>��쵿</NAME>
			<GENDER>female</GENDER>
			<AGE>21</AGE>
		</User>

		[���� ������Ʈ���� �غ��ƾ� �� ��]

		�������� �̿� ���� Ŭ���̾�Ʈ ��û�� �����ϱ� ���ؼ� ���������� �����ϴ� ����� �����ؾ� ��
		System::getSystemContents().getDatabaseManager(); �ڵ带 ���� DatabaseManager�� ���� ����
		DatabaseManager�� isLoaded() �� load() �Լ��� ���� �ε� ���� �� �ε� ���� ����
		findId(id) �Լ��� ���� id�� �����ϴ��� ���� ����
		getUserInfo(id) �Լ��� ���� UserInfo�� ���� �� ����.
		�Ʒ��� �ڵ忡�� DatabaseManager ��뿡 ���� ��Ʈ�� ���� �� ����.
		�� Ŭ���̾�Ʈ�� ������ DatabaseManager Ŭ������ ���� �ٸ� �������̽���
		�����ϹǷ� �����Ǵ� ��ü ����� �Լ����� ���� ���캼 ��

		user1, user2 ��� �׽�Ʈ �غ� ��.
	*/

	DatabaseManager &dbManger = System::getSystemContents().getDatabaseManager();
	ClientNetwork &network = System::getSystemContents().getClientNetwork();

	// �ε��� �Ǿ� ���� ������ �ε� �õ�. �ε� ���� �� false ��ȯ
	if (!dbManger.isLoaded())
		if (dbManger.load(Constant::Database::DB_ROOT, false))
			return AuthorizingResult::FAILED_DB_ERROR;

	// DatabaseManager�κ��� ip �� port ���� ����.
	const tstring SERVER_IP =
		dbManger.getAttribute(ConfigSectionType::NETWORK, _T("server_ip"));

	const int SERVER_PORT =
		Parser::tstring$Int(dbManger.getAttribute(ConfigSectionType::NETWORK, _T("server_port")));

	// TO-DO: Network logics
	UserInfo userInfoFromNetworkServer =
	{
		_T("���̵�"),
		_T("��й�ȣ"),
		_T("�̸�"),
		_T("����"),
		_T("����")
	};

	this->__userInfo = userInfoFromNetworkServer;

	// UserInfo�� �Ѱܹ��� �� Debugger ����� ���� Ȯ��
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