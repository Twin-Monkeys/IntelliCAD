// CCloudServiceDlg.cpp: 구현 파일
//

#include "stdafx.h"
#include "IntelliCAD.h"
#include "CCloudServiceDlg.h"
#include "afxdialogex.h"


// CCloudServiceDlg 대화 상자

IMPLEMENT_DYNAMIC(CCloudServiceDlg, CDialogEx)

CCloudServiceDlg::CCloudServiceDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG_CLOUD_SERVICE, pParent)
{

}

CCloudServiceDlg::~CCloudServiceDlg()
{
}

void CCloudServiceDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CCloudServiceDlg, CDialogEx)
END_MESSAGE_MAP()


// CCloudServiceDlg 메시지 처리기
