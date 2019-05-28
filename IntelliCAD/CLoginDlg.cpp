// CLoginDlg.cpp: 구현 파일
//

#include "stdafx.h"
#include "IntelliCAD.h"
#include "CLoginDlg.h"
#include "tstring.h"
#include "Parser.hpp"
#include "System.h"

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
END_MESSAGE_MAP()


// CLoginDlg 메시지 처리기

BOOL CLoginDlg::PreTranslateMessage(MSG* pMsg)
{
	// Enter, ESC key를 누르면 프로그램이 종료되는 것을 막는다.
	if (pMsg->wParam == VK_RETURN || pMsg->wParam == VK_ESCAPE)
		return TRUE;

	return CDialogEx::PreTranslateMessage(pMsg);
}

void CLoginDlg::OnBnClickedButtonSignIn()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	CString tmp;
	__ddxc_id.GetWindowText(tmp);
	tstring id = Parser::CString$tstring(tmp);

	__ddxc_pw.GetWindowText(tmp);
	tstring pw = Parser::CString$tstring(tmp);

	RemoteAccessAuthorizer &accessAuthorizer =
		System::getSystemContents().getRemoteAccessAuthorizer();

	if (accessAuthorizer.authorize(id, pw))
		AfxMessageBox(_T("승인"));
	else
		AfxMessageBox(_T("접근 불가"));
	
	EndDialog(0);
}
