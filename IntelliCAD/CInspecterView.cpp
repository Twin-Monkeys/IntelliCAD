#include "stdafx.h"
#include "IntelliCAD.h"
#include "CInspecterView.h"
#include "Constant.h"
#include "Parser.hpp"
#include "System.h"

using namespace std;

IMPLEMENT_DYNCREATE(CInspecterView, CFormView)

BEGIN_MESSAGE_MAP(CInspecterView, CFormView)
	ON_WM_SIZE()
	ON_NOTIFY(TCN_SELCHANGE, IDC_INSPECTOR_TAB, &CInspecterView::OnTcnSelchangeInspectorTab)
	ON_WM_CREATE()
END_MESSAGE_MAP()

CInspecterView::CInspecterView() : 
	CFormView(IDD_DIALOG_INSPECTOR)
{
}

CInspecterView::~CInspecterView()
{
}

void CInspecterView::DoDataExchange(CDataExchange* pDX)
{
	CFormView::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_INSPECTOR_TAB, __ddx_tab);
}

void CInspecterView::__recalcLayout()
{
	CRect tabRect;
	__ddx_tab.GetClientRect(tabRect);

	const int x = 4;
	const int y = 24;
	const int cx = (tabRect.Width() - 10);
	const int cy = (tabRect.Height() - 30);

	// 탭 추가시 여기도 추가
	__imageInfoDlg.SetWindowPos(
		nullptr, x, y, cx, cy, SWP_NOZORDER);

	__serverActiveDlg.SetWindowPos(
		nullptr, x, y, cx, cy, SWP_NOZORDER);

	__serverInactiveDlg.SetWindowPos(
		nullptr, x, y, cx, cy, SWP_NOZORDER);

	__logDlg.SetWindowPos(
		nullptr, x, y, cx, cy, SWP_NOZORDER);
}

void CInspecterView::__updateTabDlg()
{
	if (__pPrevTabDlg)
		__pPrevTabDlg->ShowWindow(SW_HIDE);

	const int TAB_IDX = __ddx_tab.GetCurSel();

	switch (TAB_IDX)
	{
	case 0:
		__pPrevTabDlg = &__imageInfoDlg;
		break;

	case 1:
		if (__serverActivated)
			__pPrevTabDlg = &__serverActiveDlg;
		else
			__pPrevTabDlg = &__serverInactiveDlg;

		break;
		
	case 2:
		__pPrevTabDlg = &__logDlg;
		break;
	}

	__pPrevTabDlg->ShowWindow(SW_SHOW);
}

void CInspecterView::OnInitialUpdate()
{
	CFormView::OnInitialUpdate();

	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.
	CString tmp;

	// tab0
	__ddx_tab.InsertItem(0, Constant::UI::TAB_NAMES[0].c_str());
	__imageInfoDlg.Create(IDD_DIALOG_IMAGE_INFO, &__ddx_tab);

	// tab1
	__ddx_tab.InsertItem(1, Constant::UI::TAB_NAMES[1].c_str());
	__serverActiveDlg.Create(IDD_DIALOG_SERVER_INTERACTION_ACTIVE, &__ddx_tab);
	__serverInactiveDlg.Create(IDD_DIALOG_SERVER_INTERACTION_INACTIVE, &__ddx_tab);

	// tab2
	__ddx_tab.InsertItem(2, Constant::UI::TAB_NAMES[2].c_str());
	__logDlg.Create(IDD_DIALOG_LOG, &__ddx_tab);

	__recalcLayout();

	__ddx_tab.SetCurSel(0);
	__updateTabDlg();

	__initialized = true;
}


void CInspecterView::OnSize(UINT nType, int cx, int cy)
{
	CFormView::OnSize(nType, cx, cy);

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
	if (!__initialized)
		return;

	__recalcLayout();
}


void CInspecterView::OnTcnSelchangeInspectorTab(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	__updateTabDlg();

	*pResult = 0;
}

void CInspecterView::onLoginSuccess(const Account &account)
{
	__serverActivated = true;
	__serverActiveDlg.setUserId(account.name);

	__updateTabDlg();
}

int CInspecterView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (__super::OnCreate(lpCreateStruct) == -1)
		return -1;

	// TODO:  여기에 특수화된 작성 코드를 추가합니다.
	System::getSystemContents().
		getEventBroadcaster().addLoginSuccessListener(*this);

	return 0;
}