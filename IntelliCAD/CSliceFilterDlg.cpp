// CSliceFilterDlg.cpp: 구현 파일
//

#include "stdafx.h"
#include "IntelliCAD.h"
#include "CSliceFilterDlg.h"
#include "afxdialogex.h"
#include "System.h"

using namespace std;

// CSliceFilterDlg 대화 상자

IMPLEMENT_DYNAMIC(CSliceFilterDlg, CDialogEx)

BEGIN_MESSAGE_MAP(CSliceFilterDlg, CDialogEx)
	ON_WM_SIZE()
	ON_WM_PAINT()
	ON_WM_GETMINMAXINFO()
	ON_CONTROL(CVN_MouseMovePlotArea, IDC_SLICE_FILTER_chartView, OnMouseMovePlotArea)
	ON_COMMAND(ID_MENU_SLICE_FILTER_EDIT_initSliceFilter, &CSliceFilterDlg::OnMenuSliceFilterEditInitSliceFilter)
	ON_WM_ERASEBKGND()
	ON_WM_DESTROY()
END_MESSAGE_MAP()

CSliceFilterDlg::CSliceFilterDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG_SLICE_FILTER, pParent), __chartBuilder(__ddx_chartViewer)
{
	System::getSystemContents().
		getEventBroadcaster().addInitSliceTransferFunctionListener(*this);
}

CSliceFilterDlg::~CSliceFilterDlg()
{
}

void CSliceFilterDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_SLICE_FILTER_chartView, __ddx_chartViewer);
}


// CSliceFilterDlg 메시지 처리기

void CSliceFilterDlg::__init()
{
	RenderingEngine::ImageProcessor &imageProcessor =
		System::getSystemContents().getRenderingEngine().imageProcessor;

	__chartBuilder.initData(imageProcessor.getTransferFunctionAs<double>());
	__activated = true;
}

void CSliceFilterDlg::__render()
{
	// TODO:  여기에 추가 초기화 작업을 추가합니다.
	CRect rect;
	GetClientRect(rect);

	__chartBuilder.renderChart(rect.Width(), rect.Height());
}

bool CSliceFilterDlg::isActive() const
{
	return __activated;
}

void CSliceFilterDlg::OnSize(UINT nType, int cx, int cy)
{
	CDialogEx::OnSize(nType, cx, cy);

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
	Invalidate();
}

void CSliceFilterDlg::OnPaint()
{
	CPaintDC dc(this); // device context for painting
					   // TODO: 여기에 메시지 처리기 코드를 추가합니다.
					   // 그리기 메시지에 대해서는 CDialogEx::OnPaint()을(를) 호출하지 마십시오.
	if (__activated)
		__render();
}

void CSliceFilterDlg::OnLButtonDownChartView()
{
	__chartViewLButtonDown = true;
}

void CSliceFilterDlg::OnLButtonUpChartView()
{
	__chartViewLButtonDown = false;
	__chartBuilder.finishDragging();
}

void CSliceFilterDlg::OnMouseMovePlotArea()
{
	const CPoint PLOT_MOUSE_POS = __chartBuilder.getMousePositionInPlot();

	// 마우스 좌측 버튼이 눌려있지 않다면
	if (!(GetKeyState(VK_LBUTTON) & 0x100))
		__chartViewLButtonDown = false;

	if (__chartViewLButtonDown)
	{
		__chartBuilder.dragPlot(PLOT_MOUSE_POS);
		__render();
	}

	__chartBuilder.renderPlotCrosshair(PLOT_MOUSE_POS);
}


BOOL CSliceFilterDlg::PreTranslateMessage(MSG* pMsg)
{
	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.
	if (TranslateAccelerator(m_hWnd, __hAccel, pMsg))
		return true;

	if (pMsg->hwnd == __ddx_chartViewer.GetSafeHwnd())
	{
		switch (pMsg->message)
		{
		case WM_LBUTTONDOWN:
			OnLButtonDownChartView();
			return true;

		case WM_LBUTTONUP:
			OnLButtonUpChartView();
			return true;
		}
	}

	return CDialogEx::PreTranslateMessage(pMsg);
}

void CSliceFilterDlg::OnGetMinMaxInfo(MINMAXINFO* lpMMI)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	lpMMI->ptMinTrackSize.x = 500;
	lpMMI->ptMinTrackSize.y = 250;

	CDialogEx::OnGetMinMaxInfo(lpMMI);
}

void CSliceFilterDlg::onInitSliceTransferFunction()
{
	__init();
	__render();
}

void CSliceFilterDlg::OnMenuSliceFilterEditInitSliceFilter()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	System::getSystemContents().
		getRenderingEngine().imageProcessor.initTransferFunction();
}


BOOL CSliceFilterDlg::OnEraseBkgnd(CDC* pDC)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	return true;
	//return __super::OnEraseBkgnd(pDC);
}

BOOL CSliceFilterDlg::OnInitDialog()
{
	__super::OnInitDialog();

	// TODO:  여기에 추가 초기화 작업을 추가합니다.
	__hAccel = LoadAccelerators(
		AfxGetResourceHandle(), MAKEINTRESOURCE(IDR_ACCELERATOR_SLICE_FILTER_DLG));

	return TRUE;  // return TRUE unless you set the focus to a control
				  // 예외: OCX 속성 페이지는 FALSE를 반환해야 합니다.
}

void CSliceFilterDlg::OnDestroy()
{
	DestroyAcceleratorTable(__hAccel);

	__super::OnDestroy();
}