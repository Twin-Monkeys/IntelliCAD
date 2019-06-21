// CServerInteractionActiveDlg.cpp: 구현 파일
//

#include "stdafx.h"
#include "IntelliCAD.h"
#include "CServerInteractionActiveDlg.h"
#include "afxdialogex.h"
#include "Parser.hpp"
#include "System.h"

using namespace std;

// CServerInteractionActiveDlg 대화 상자

IMPLEMENT_DYNAMIC(CServerInteractionActiveDlg, CDialogEx)

BEGIN_MESSAGE_MAP(CServerInteractionActiveDlg, CDialogEx)
	ON_BN_CLICKED(IDC_SERVER_ACTIVE_BUTTON_imgDownTest, &CServerInteractionActiveDlg::OnBnClickedServerActiveButtonImgDownTest)
END_MESSAGE_MAP()

CServerInteractionActiveDlg::CServerInteractionActiveDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG_SERVER_INTERACTION_ACTIVE, pParent)
{

}

CServerInteractionActiveDlg::~CServerInteractionActiveDlg()
{
}

void CServerInteractionActiveDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_SERVER_ACTIVE_userId, __ddx_userId);
}

// CServerInteractionActiveDlg 메시지 처리기

void CServerInteractionActiveDlg::setUserId(const tstring &id)
{
	__ddx_userId.SetWindowText(id.c_str());
}

void CServerInteractionActiveDlg::OnBnClickedServerActiveButtonImgDownTest()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	System::getSystemContents().getRemoteAccessAuthorizer().requestServerDBFile(
		ServerFileDBSectionType::CT_IMAGE,
			_T("1.3.6.1.4.1.14519.5.2.1.6279.6001.970264865033574190975654369557.mhd"));

	System::getSystemContents().getRemoteAccessAuthorizer().requestServerDBFile(
		ServerFileDBSectionType::CT_IMAGE,
		_T("1.3.6.1.4.1.14519.5.2.1.6279.6001.970264865033574190975654369557.raw"));
}
