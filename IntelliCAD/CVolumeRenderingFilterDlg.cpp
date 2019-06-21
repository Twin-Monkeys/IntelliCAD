// CVolumeRenderingFilterDlg.cpp: 구현 파일
//

#include "stdafx.h"
#include "IntelliCAD.h"
#include "CVolumeRenderingFilterDlg.h"
#include "afxdialogex.h"
#include "System.h"

// CVolumeRenderingFilterDlg 대화 상자

IMPLEMENT_DYNAMIC(CVolumeRenderingFilterDlg, CDialogEx)

BEGIN_MESSAGE_MAP(CVolumeRenderingFilterDlg, CDialogEx)
	ON_WM_GETMINMAXINFO()
	ON_WM_PAINT()
	ON_WM_SIZE()
	ON_CONTROL(CVN_MouseMovePlotArea, IDC_VOLUME_RENDERING_FILTER_chartView, OnMouseMovePlotArea)
	ON_COMMAND(ID_MENU_VOLUME_RENDERING_FILTER_EDIT_initRedFilter, &CVolumeRenderingFilterDlg::OnMenuInitRedFilter)
	ON_COMMAND(ID_MENU_VOLUME_RENDERING_FILTER_EDIT_initGreenFilter, &CVolumeRenderingFilterDlg::OnMenuInitGreenFilter)
	ON_COMMAND(ID_MENU_VOLUME_RENDERING_FILTER_EDIT_initBlueFilter, &CVolumeRenderingFilterDlg::OnMenuInitBlueFilter)
	ON_COMMAND(ID_MENU_VOLUME_RENDERING_FILTER_EDIT_initAlphaFilter, &CVolumeRenderingFilterDlg::OnMenuInitAlphaFilter)
	ON_COMMAND(ID_MENU_VOLUME_RENDERING_FILTER_EDIT_initAllFilter, &CVolumeRenderingFilterDlg::OnMenuInitAllFilter)
	ON_COMMAND(ID_MENU_VOLUME_RENDERING_FILTER_EDIT_toggleTargetFilterRed, &CVolumeRenderingFilterDlg::OnMenuToggleTargetFilterRed)
	ON_COMMAND(ID_MENU_VOLUME_RENDERING_FILTER_EDIT_toggleTargetFilterGreen, &CVolumeRenderingFilterDlg::OnMenuToggleTargetFilterGreen)
	ON_COMMAND(ID_MENU_VOLUME_RENDERING_FILTER_EDIT_toggleTargetFilterBlue, &CVolumeRenderingFilterDlg::OnMenuToggleTargetFilterBlue)
	ON_COMMAND(ID_MENU_VOLUME_RENDERING_FILTER_EDIT_toggleTargetFilterAlpha, &CVolumeRenderingFilterDlg::OnMenuToggleTargetFilterAlpha)
	ON_COMMAND(ID_MENU_VOLUME_RENDERING_FILTER_EDIT_selectAllTargetFilter, &CVolumeRenderingFilterDlg::OnMenuSelectAllTargetFilter)
	ON_COMMAND(ID_MENU_VOLUME_RENDERING_FILTER_EDIT_selectNoneTargetFilter, &CVolumeRenderingFilterDlg::OnMenuSelectNoneTargetFilter)
	ON_WM_ERASEBKGND()
	ON_WM_DESTROY()
END_MESSAGE_MAP()

CVolumeRenderingFilterDlg::CVolumeRenderingFilterDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG_VOLUME_RENDERING_FILTER, pParent), __chartBuilder(__ddx_chartViewer)
{
	System::getSystemContents().
		getEventBroadcaster().addInitVolumeTransferFunctionListener(*this);
}

CVolumeRenderingFilterDlg::~CVolumeRenderingFilterDlg()
{
}

void CVolumeRenderingFilterDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_VOLUME_RENDERING_FILTER_chartView, __ddx_chartViewer);
}

// CVolumeRenderingFilterDlg 메시지 처리기

void CVolumeRenderingFilterDlg::__init(const ColorChannelType colorType)
{
	RenderingEngine::VolumeRenderer &volumeRenderer =
		System::getSystemContents().getRenderingEngine().volumeRenderer;

	if (colorType == ColorChannelType::ALL)
	{
		__chartBuilder.initData(
			volumeRenderer.getTransferFunctionAs<double>(ColorChannelType::RED),
			volumeRenderer.getTransferFunctionAs<double>(ColorChannelType::GREEN),
			volumeRenderer.getTransferFunctionAs<double>(ColorChannelType::BLUE),
			volumeRenderer.getTransferFunctionAs<double>(ColorChannelType::ALPHA));
	}
	else
		__chartBuilder.initData(colorType, volumeRenderer.getTransferFunctionAs<double>(colorType));

	__activated = true;
}

void CVolumeRenderingFilterDlg::__render()
{
	// TODO:  여기에 추가 초기화 작업을 추가합니다.
	CRect rect;
	GetClientRect(rect);

	__chartBuilder.renderChart(rect.Width(), rect.Height());
}

void CVolumeRenderingFilterDlg::__setMenuItemChecked(const UINT nIDCheckItem, const bool checked)
{
	const UINT CHECK_FLAG = (checked ? MF_CHECKED : MF_UNCHECKED);
	GetMenu()->CheckMenuItem(nIDCheckItem, CHECK_FLAG | MF_BYCOMMAND);
}

bool CVolumeRenderingFilterDlg::isActive() const
{
	return __activated;
}

void CVolumeRenderingFilterDlg::OnSize(UINT nType, int cx, int cy)
{
	__super::OnSize(nType, cx, cy);

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
	Invalidate();
}

void CVolumeRenderingFilterDlg::OnPaint()
{
	CPaintDC dc(this); // device context for painting
						   // TODO: 여기에 메시지 처리기 코드를 추가합니다.
						   // 그리기 메시지에 대해서는 CDialogEx::OnPaint()을(를) 호출하지 마십시오.
	if (__activated)
		__render();
}

void CVolumeRenderingFilterDlg::OnLButtonDownChartView()
{
	__chartViewLButtonDown = true;
}

void CVolumeRenderingFilterDlg::OnLButtonUpChartView()
{
	__chartViewLButtonDown = false;
	__chartBuilder.finishDragging();
}

void CVolumeRenderingFilterDlg::OnMouseMovePlotArea()
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

BOOL CVolumeRenderingFilterDlg::PreTranslateMessage(MSG* pMsg)
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

	return __super::PreTranslateMessage(pMsg);
}

void CVolumeRenderingFilterDlg::OnGetMinMaxInfo(MINMAXINFO* lpMMI)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	lpMMI->ptMinTrackSize.x = 500;
	lpMMI->ptMinTrackSize.y = 250;

	CDialogEx::OnGetMinMaxInfo(lpMMI);
}

void CVolumeRenderingFilterDlg::onInitVolumeTransferFunction(const ColorChannelType colorType)
{
	__init(colorType);
	__render();
}

void CVolumeRenderingFilterDlg::OnMenuInitRedFilter()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	System::getSystemContents().
		getRenderingEngine().volumeRenderer.initTransferFunction(ColorChannelType::RED);
}


void CVolumeRenderingFilterDlg::OnMenuInitGreenFilter()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	System::getSystemContents().
		getRenderingEngine().volumeRenderer.initTransferFunction(ColorChannelType::GREEN);
}


void CVolumeRenderingFilterDlg::OnMenuInitBlueFilter()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	System::getSystemContents().
		getRenderingEngine().volumeRenderer.initTransferFunction(ColorChannelType::BLUE);
}

void CVolumeRenderingFilterDlg::OnMenuInitAlphaFilter()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	System::getSystemContents().
		getRenderingEngine().volumeRenderer.initTransferFunction(ColorChannelType::ALPHA);
}

void CVolumeRenderingFilterDlg::OnMenuInitAllFilter()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	System::getSystemContents().
		getRenderingEngine().volumeRenderer.initTransferFunction(ColorChannelType::ALL);
}


void CVolumeRenderingFilterDlg::OnMenuToggleTargetFilterRed()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	const ColorChannelType TARGET_COLOR = ColorChannelType::RED;

	__chartBuilder.toggleColorActivation(TARGET_COLOR);
	
	__setMenuItemChecked(
		ID_MENU_VOLUME_RENDERING_FILTER_EDIT_toggleTargetFilterRed,
		__chartBuilder.activeColorFlagArr[TARGET_COLOR]);
}

void CVolumeRenderingFilterDlg::OnMenuToggleTargetFilterGreen()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	const ColorChannelType TARGET_COLOR = ColorChannelType::GREEN;

	__chartBuilder.toggleColorActivation(TARGET_COLOR);

	__setMenuItemChecked(
		ID_MENU_VOLUME_RENDERING_FILTER_EDIT_toggleTargetFilterGreen,
		__chartBuilder.activeColorFlagArr[TARGET_COLOR]);
}

void CVolumeRenderingFilterDlg::OnMenuToggleTargetFilterBlue()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	const ColorChannelType TARGET_COLOR = ColorChannelType::BLUE;

	__chartBuilder.toggleColorActivation(TARGET_COLOR);

	__setMenuItemChecked(
		ID_MENU_VOLUME_RENDERING_FILTER_EDIT_toggleTargetFilterBlue,
		__chartBuilder.activeColorFlagArr[TARGET_COLOR]);
}

void CVolumeRenderingFilterDlg::OnMenuToggleTargetFilterAlpha()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	const ColorChannelType TARGET_COLOR = ColorChannelType::ALPHA;

	__chartBuilder.toggleColorActivation(TARGET_COLOR);

	__setMenuItemChecked(
		ID_MENU_VOLUME_RENDERING_FILTER_EDIT_toggleTargetFilterAlpha,
		__chartBuilder.activeColorFlagArr[TARGET_COLOR]);
}

void CVolumeRenderingFilterDlg::OnMenuSelectAllTargetFilter()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	__chartBuilder.setColorActivation(ColorChannelType::ALL, true);

	__setMenuItemChecked(
		ID_MENU_VOLUME_RENDERING_FILTER_EDIT_toggleTargetFilterRed, true);

	__setMenuItemChecked(
		ID_MENU_VOLUME_RENDERING_FILTER_EDIT_toggleTargetFilterGreen, true);

	__setMenuItemChecked(
		ID_MENU_VOLUME_RENDERING_FILTER_EDIT_toggleTargetFilterBlue, true);

	__setMenuItemChecked(
		ID_MENU_VOLUME_RENDERING_FILTER_EDIT_toggleTargetFilterAlpha, true);
}

void CVolumeRenderingFilterDlg::OnMenuSelectNoneTargetFilter()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	__chartBuilder.setColorActivation(ColorChannelType::ALL, false);

	__setMenuItemChecked(
		ID_MENU_VOLUME_RENDERING_FILTER_EDIT_toggleTargetFilterRed, false);

	__setMenuItemChecked(
		ID_MENU_VOLUME_RENDERING_FILTER_EDIT_toggleTargetFilterGreen, false);

	__setMenuItemChecked(
		ID_MENU_VOLUME_RENDERING_FILTER_EDIT_toggleTargetFilterBlue, false);

	__setMenuItemChecked(
		ID_MENU_VOLUME_RENDERING_FILTER_EDIT_toggleTargetFilterAlpha, false);
}


BOOL CVolumeRenderingFilterDlg::OnEraseBkgnd(CDC* pDC)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	return true;
	// return __super::OnEraseBkgnd(pDC);
}


BOOL CVolumeRenderingFilterDlg::OnInitDialog()
{
	__super::OnInitDialog();

	// TODO:  여기에 추가 초기화 작업을 추가합니다.
	__hAccel = LoadAccelerators(
		AfxGetResourceHandle(), MAKEINTRESOURCE(IDR_ACCELERATOR_VOLUME_RENDERING_FILTER_DLG));

	return TRUE;  // return TRUE unless you set the focus to a control
				  // 예외: OCX 속성 페이지는 FALSE를 반환해야 합니다.
}

void CVolumeRenderingFilterDlg::OnDestroy()
{
	DestroyAcceleratorTable(__hAccel);

	__super::OnDestroy();
}