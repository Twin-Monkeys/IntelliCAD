// CLoginDlg.cpp: 구현 파일
//

#include "stdafx.h"
#include "IntelliCAD.h"
#include "CLoginDlg.h"
#include "afxdialogex.h"
#include "Constant.h"

// CLoginDlg 대화 상자

IMPLEMENT_DYNAMIC(CLoginDlg, CDialogEx)

CLoginDlg::CLoginDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG2, pParent)
{
	__loadLogo();
}

CLoginDlg::~CLoginDlg()
{
}

void CLoginDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CLoginDlg, CDialogEx)
	ON_BN_CLICKED(IDC_BUTTON1, &CLoginDlg::OnBnClickedButton1)
	ON_WM_PAINT()
END_MESSAGE_MAP()


// CLoginDlg 메시지 처리기

void CLoginDlg::OnBnClickedButton1()
{
	EndDialog(SIGN_IN);
}

void CLoginDlg::OnPaint()
{
	CPaintDC dc(this); 
	
	::SetStretchBltMode(dc.m_hDC, COLORONCOLOR);
	__logo.StretchBlt(dc.m_hDC, 190, 0, 100, 150, SRCCOPY);
}

void CLoginDlg::__loadLogo() 
{
	__logo.Destroy();
	__logo.Load(TEXT("res/logo.png"));
}

BOOL CLoginDlg::PreTranslateMessage(MSG* pMsg)
{
	// Enter, ESC key를 누르면 프로그램이 종료되는 것을 막는다.
	if (pMsg->wParam == VK_RETURN || pMsg->wParam == VK_ESCAPE)
		return TRUE;

	return CDialogEx::PreTranslateMessage(pMsg);
}
