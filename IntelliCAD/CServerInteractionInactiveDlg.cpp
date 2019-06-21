// CServerInteractionInactiveDlg.cpp: 구현 파일
//

#include "stdafx.h"
#include "IntelliCAD.h"
#include "CServerInteractionInactiveDlg.h"
#include "afxdialogex.h"


// CServerInteractionInactiveDlg 대화 상자

IMPLEMENT_DYNAMIC(CServerInteractionInactiveDlg, CDialogEx)

BEGIN_MESSAGE_MAP(CServerInteractionInactiveDlg, CDialogEx)
END_MESSAGE_MAP()

CServerInteractionInactiveDlg::CServerInteractionInactiveDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG_SERVER_INTERACTION_INACTIVE, pParent)
{

}

CServerInteractionInactiveDlg::~CServerInteractionInactiveDlg()
{
}

void CServerInteractionInactiveDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


// CServerInteractionInactiveDlg 메시지 처리기
