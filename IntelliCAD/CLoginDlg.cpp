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

BOOL CLoginDlg::PreTranslateMessage(MSG* pMsg)
{
	// Enter, ESC key를 누르면 프로그램이 종료되는 것을 막는다.
	if ((pMsg->wParam == VK_RETURN) || (pMsg->wParam == VK_ESCAPE))
		return TRUE;

	return CDialogEx::PreTranslateMessage(pMsg);
}

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

	switch (accessAuthorizer.authorize(ID, PASSWORD))
	{
	case AuthorizingResult::FAILED_DB_ERROR:
		Debugger::popMessageBox(_T("Authorize 결과: FAILED_DB_ERROR"));
		break;

	case AuthorizingResult::FAILED_INVALID_ID:
		Debugger::popMessageBox(_T("Authorize 결과: FAILED_INVALID_ID"));
		break;

	case AuthorizingResult::FAILED_NETWORK_ERROR:
		Debugger::popMessageBox(_T("Authorize 결과: FAILED_NETWORK_ERROR"));
		break;

	case AuthorizingResult::FAILED_WRONG_PASSWORD:
		Debugger::popMessageBox(_T("Authorize 결과: FAILED_WRONG_PASSWORD"));
		break;

	case AuthorizingResult::SUCCESS:
		Debugger::popMessageBox(_T("Authorize 결과: SUCCESS"));
		break;
	}
	
	EndDialog(0);
}


void CLoginDlg::OnBnClickedButtonClose()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	EndDialog(IDCLOSE);
}
