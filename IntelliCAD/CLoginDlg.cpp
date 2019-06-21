// CLoginDlg.cpp: 구현 파일
//

#include "stdafx.h"
#include "IntelliCAD.h"
#include "CLoginDlg.h"
#include "tstring.h"
#include "Parser.hpp"
#include "System.h"
#include "Debugger.h"

using namespace std;

// CLoginDlg 대화 상자

IMPLEMENT_DYNAMIC(CLoginDlg, CDialogEx)

CLoginDlg::CLoginDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG_LOGIN, pParent)
{
}

CLoginDlg::~CLoginDlg()
{
}

void CLoginDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_LOGINDLG_EDIT_ID, __ddxc_id);
	DDX_Control(pDX, IDC_LOGINDLG_EDIT_PW, __ddxc_pw);
}


BEGIN_MESSAGE_MAP(CLoginDlg, CDialogEx)
	ON_BN_CLICKED(IDC_LOGINDLG_BUTTON_SIGN_IN, &CLoginDlg::OnBnClickedButtonSignIn)
	ON_BN_CLICKED(IDC_BUTTON_CLOSE, &CLoginDlg::OnBnClickedButtonClose)
END_MESSAGE_MAP()


// CLoginDlg 메시지 처리기

void CLoginDlg::OnBnClickedButtonSignIn()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	CString tmp;
	__ddxc_id.GetWindowText(tmp);
	const tstring ID = Parser::CString$tstring(tmp);

	if (ID.empty())
	{
		MessageBox(_T("아이디를 입력해주세요."), _T("로그인 오류"), MB_ICONEXCLAMATION);
		__ddxc_id.SetFocus();
		return;
	}

	__ddxc_pw.GetWindowText(tmp);
	const tstring PASSWORD = Parser::CString$tstring(tmp);

	if (PASSWORD.empty())
	{
		MessageBox(_T("비밀번호를 입력해주세요."), _T("로그인 오류"), MB_ICONEXCLAMATION);
		__ddxc_pw.SetFocus();
		return;
	}

	RemoteAccessAuthorizer &accessAuthorizer =
		System::getSystemContents().getRemoteAccessAuthorizer();

	AuthorizingResult result;

	bool loop;
	do
	{
		loop = false;
		result = accessAuthorizer.authorize(ID, PASSWORD);

		switch (result)
		{
		case AuthorizingResult::FAILED_DB_ERROR:
		{
			const int MB_RET =
				MessageBox(
					_T("설정 DB 정보를 불러올 수 없습니다."), _T("로그인 오류"),
					MB_RETRYCANCEL | MB_ICONEXCLAMATION);

			if (MB_RET == IDRETRY)
				loop = true;
		}
			break;

		case AuthorizingResult::FAILED_NETWORK_ERROR:
		{
			const int MB_RET =
				MessageBox(
					_T("서버 접속에 실패하였습니다."), _T("로그인 오류"),
					MB_RETRYCANCEL | MB_ICONEXCLAMATION);

			if (MB_RET == IDRETRY)
				loop = true;
		}
			break;

		case AuthorizingResult::FAILED_INVALID_ID:
			MessageBox(_T("존재하지 않는 ID입니다."), _T("로그인 오류"), MB_ICONEXCLAMATION);
			break;

		case AuthorizingResult::FAILED_WRONG_PASSWORD:
			MessageBox(_T("비밀번호가 올바르지 않습니다."), _T("로그인 오류"), MB_ICONEXCLAMATION);
			break;
		}
	}
	while (loop);

	if (result == AuthorizingResult::SUCCESS)
		EndDialog(LOGIN_DLG_LOGIN_SUCCESS);
}


void CLoginDlg::OnBnClickedButtonClose()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	EndDialog(IDCLOSE);
}


void CLoginDlg::OnOK()
{
	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.

	// CDialogEx::OnOK();
}


void CLoginDlg::OnCancel()
{
	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.

	CDialogEx::OnCancel();
}
