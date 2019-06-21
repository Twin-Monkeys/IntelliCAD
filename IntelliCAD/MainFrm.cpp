// 이 MFC 샘플 소스 코드는 MFC Microsoft Office Fluent 사용자 인터페이스("Fluent UI")를
// 사용하는 방법을 보여 주며, MFC C++ 라이브러리 소프트웨어에 포함된
// Microsoft Foundation Classes Reference 및 관련 전자 문서에 대해
// 추가적으로 제공되는 내용입니다.
// Fluent UI를 복사, 사용 또는 배포하는 데 대한 사용 약관은 별도로 제공됩니다.
// Fluent UI 라이선싱 프로그램에 대한 자세한 내용은
// https://go.microsoft.com/fwlink/?LinkId=238214.
//
// Copyright (C) Microsoft Corporation
// All rights reserved.

// MainFrm.cpp: CMainFrame 클래스의 구현
//

#include "stdafx.h"
#include "IntelliCAD.h"
#include "MainFrm.h"
#include "IntelliCADView.h"
#include "CVolumeRenderingView.h"
#include "CInspecterView.h"
#include "tstring.h"
#include "Parser.hpp"
#include "System.h"
#include "VolumeReader.h"
#include "CSliceView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CMainFrame

using namespace std;

IMPLEMENT_DYNCREATE(CMainFrame, CFrameWndEx)

BEGIN_MESSAGE_MAP(CMainFrame, CFrameWndEx)
	ON_WM_CREATE()
	ON_COMMAND(ID_FILE_OPEN, &CMainFrame::OnFileOpen)
	ON_COMMAND(ID_BUTTON_CLOUD, &CMainFrame::OnButtonCloudService)
	ON_COMMAND_RANGE(ID_FILE_MRU_FILE1, ID_FILE_MRU_FILE4, OnOpenMRUFile)

	// 단면도 탭
	ON_COMMAND(ID_MAIN_RIBBON_BUTTON_setSliceFilter, &CMainFrame::OnMainRibbonButtonSetSliceFilter)
	ON_COMMAND(ID_MAIN_RIBBON_BUTTON_initSliceFilter, &CMainFrame::OnMainRibbonButtonInitSliceFilter)

	ON_COMMAND(ID_MAIN_RIBBON_EDIT_slicingPointX, &CMainFrame::OnMainRibbonUpdateSlicingPoint)
	ON_COMMAND(ID_MAIN_RIBBON_EDIT_slicingPointY, &CMainFrame::OnMainRibbonUpdateSlicingPoint)
	ON_COMMAND(ID_MAIN_RIBBON_EDIT_slicingPointZ, &CMainFrame::OnMainRibbonUpdateSlicingPoint)
	ON_COMMAND(ID_MAIN_RIBBON_BUTTON_initSlicinigPoint, &CMainFrame::OnMainRibbonInitSlicinigPoint)

	ON_COMMAND(ID_MAIN_RIBBON_EDIT_anchorAdjX_top, &CMainFrame::OnMainRibbonUpdateAnchorAdj_top)
	ON_COMMAND(ID_MAIN_RIBBON_EDIT_anchorAdjY_top, &CMainFrame::OnMainRibbonUpdateAnchorAdj_top)
	ON_COMMAND(ID_MAIN_RIBBON_BUTTON_initAnchorAdj_top, &CMainFrame::OnMainRibbonInitAnchorAdj_top)

	ON_COMMAND(ID_MAIN_RIBBON_EDIT_anchorAdjX_front, &CMainFrame::OnMainRibbonUpdateAnchorAdj_front)
	ON_COMMAND(ID_MAIN_RIBBON_EDIT_anchorAdjY_front, &CMainFrame::OnMainRibbonUpdateAnchorAdj_front)
	ON_COMMAND(ID_MAIN_RIBBON_BUTTON_initAnchorAdj_front, &CMainFrame::OnMainRibbonInitAnchorAdj_front)

	ON_COMMAND(ID_MAIN_RIBBON_EDIT_anchorAdjX_right, &CMainFrame::OnMainRibbonUpdateAnchorAdj_right)
	ON_COMMAND(ID_MAIN_RIBBON_EDIT_anchorAdjY_right, &CMainFrame::OnMainRibbonUpdateAnchorAdj_right)
	ON_COMMAND(ID_MAIN_RIBBON_BUTTON_initAnchorAdj_right, &CMainFrame::OnMainRibbonInitAnchorAdj_right)

	// 볼륨 렌더링 탭
	ON_COMMAND(ID_MAIN_RIBBON_BUTTON_setVolumeFilter, &CMainFrame::OnMainRibbonSetVolumeFilter)
	ON_COMMAND(ID_MAIN_RIBBON_BUTTON_initVolumeFilterRed, &CMainFrame::OnMainRibbonInitVolumeFilterRed)
	ON_COMMAND(ID_MAIN_RIBBON_BUTTON_initVolumeFilterGreen, &CMainFrame::OnMainRibbonInitVolumeFilterGreen)
	ON_COMMAND(ID_MAIN_RIBBON_BUTTON_initVolumeFilterBlue, &CMainFrame::OnMainRibbonInitVolumeFilterBlue)
	ON_COMMAND(ID_MAIN_RIBBON_BUTTON_initVolumeFilterAlpha, &CMainFrame::OnMainRibbonInitVolumeFilterAlpha)
	ON_COMMAND(ID_MAIN_RIBBON_BUTTON_initVolumeFilterAll, &CMainFrame::OnMainRibbonInitVolumeFilterAll)

	// Light 1
	ON_COMMAND(ID_MAIN_RIBBON_BUTTON_togglelLight1, &CMainFrame::OnMainRibbonButtonToggleLight1)
	ON_COMMAND(ID_MAIN_RIBBON_COLOR_BUTTON_setLight1AmbientColor, &CMainFrame::OnMainRibbonButtonSetLight1AmbientColor)
	ON_COMMAND(ID_MAIN_RIBBON_COLOR_BUTTON_setLight1DiffuseColor, &CMainFrame::OnMainRibbonButtonSetLight1DiffuseColor)
	ON_COMMAND(ID_MAIN_RIBBON_COLOR_BUTTON_setLight1SpecularColor, &CMainFrame::OnMainRibbonButtonSetLight1SpecularColor)
	ON_COMMAND(ID_MAIN_RIBBON_EDIT_setLight1XPos, &CMainFrame::OnMainRibbonButtonSetLight1XPos)
	ON_COMMAND(ID_MAIN_RIBBON_EDIT_setLight1YPos, &CMainFrame::OnMainRibbonButtonSetLight1YPos)
	ON_COMMAND(ID_MAIN_RIBBON_EDIT_setLight1ZPos, &CMainFrame::OnMainRibbonButtonSetLight1ZPos)

	// Light 2
	ON_COMMAND(ID_MAIN_RIBBON_BUTTON_togglelLight2, &CMainFrame::OnMainRibbonButtonToggleLight2)
	ON_COMMAND(ID_MAIN_RIBBON_COLOR_BUTTON_setLight2AmbientColor, &CMainFrame::OnMainRibbonButtonSetLight2AmbientColor)
	ON_COMMAND(ID_MAIN_RIBBON_COLOR_BUTTON_setLight2DiffuseColor, &CMainFrame::OnMainRibbonButtonSetLight2DiffuseColor)
	ON_COMMAND(ID_MAIN_RIBBON_COLOR_BUTTON_setLight2SpecularColor, &CMainFrame::OnMainRibbonButtonSetLight2SpecularColor)
	ON_COMMAND(ID_MAIN_RIBBON_EDIT_setLight2XPos, &CMainFrame::OnMainRibbonButtonSetLight2XPos)
	ON_COMMAND(ID_MAIN_RIBBON_EDIT_setLight2YPos, &CMainFrame::OnMainRibbonButtonSetLight2YPos)
	ON_COMMAND(ID_MAIN_RIBBON_EDIT_setLight2ZPos, &CMainFrame::OnMainRibbonButtonSetLight2ZPos)

	// Light 3
	ON_COMMAND(ID_MAIN_RIBBON_BUTTON_togglelLight3, &CMainFrame::OnMainRibbonButtonToggleLight3)
	ON_COMMAND(ID_MAIN_RIBBON_COLOR_BUTTON_setLight3AmbientColor, &CMainFrame::OnMainRibbonButtonSetLight3AmbientColor)
	ON_COMMAND(ID_MAIN_RIBBON_COLOR_BUTTON_setLight3DiffuseColor, &CMainFrame::OnMainRibbonButtonSetLight3DiffuseColor)
	ON_COMMAND(ID_MAIN_RIBBON_COLOR_BUTTON_setLight3SpecularColor, &CMainFrame::OnMainRibbonButtonSetLight3SpecularColor)
	ON_COMMAND(ID_MAIN_RIBBON_EDIT_setLight3XPos, &CMainFrame::OnMainRibbonButtonSetLight3XPos)
	ON_COMMAND(ID_MAIN_RIBBON_EDIT_setLight3YPos, &CMainFrame::OnMainRibbonButtonSetLight3YPos)
	ON_COMMAND(ID_MAIN_RIBBON_EDIT_setLight3ZPos, &CMainFrame::OnMainRibbonButtonSetLight3ZPos)

	// 진단 탭
	ON_COMMAND(ID_MAIN_RIBBON_BUTTON_analysis, &CMainFrame::OnMainRibbonAnalysis)


	// 활성화 / 비활성화

	// 단면도 탭
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_BUTTON_setSliceFilter, &CMainFrame::OnUpdateMainRibbonItem)

	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_slicingPointX, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_slicingPointY, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_slicingPointZ, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_BUTTON_initSlicinigPoint, &CMainFrame::OnUpdateMainRibbonItem)

	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_anchorAdjX_top, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_anchorAdjY_top, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_BUTTON_initAnchorAdj_top, &CMainFrame::OnUpdateMainRibbonItem)

	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_anchorAdjX_front, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_anchorAdjY_front, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_BUTTON_initAnchorAdj_front, &CMainFrame::OnUpdateMainRibbonItem)

	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_anchorAdjX_right, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_anchorAdjY_right, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_BUTTON_initAnchorAdj_right, &CMainFrame::OnUpdateMainRibbonItem)

	// 볼륨 렌더링 탭
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_BUTTON_setVolumeFilter, &CMainFrame::OnUpdateMainRibbonItem)

	// Light 1
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_BUTTON_togglelLight1, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_COLOR_BUTTON_setLight1AmbientColor, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_COLOR_BUTTON_setLight1DiffuseColor, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_COLOR_BUTTON_setLight1SpecularColor, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_setLight1XPos, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_setLight1YPos, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_setLight1ZPos, &CMainFrame::OnUpdateMainRibbonItem)

	// Light 2
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_BUTTON_togglelLight2, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_COLOR_BUTTON_setLight2AmbientColor, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_COLOR_BUTTON_setLight2DiffuseColor, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_COLOR_BUTTON_setLight2SpecularColor, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_setLight2XPos, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_setLight2YPos, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_setLight2ZPos, &CMainFrame::OnUpdateMainRibbonItem)

	// Light 3
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_BUTTON_togglelLight3, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_COLOR_BUTTON_setLight3AmbientColor, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_COLOR_BUTTON_setLight3DiffuseColor, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_COLOR_BUTTON_setLight3SpecularColor, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_setLight3XPos, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_setLight3YPos, &CMainFrame::OnUpdateMainRibbonItem)
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_EDIT_setLight3ZPos, &CMainFrame::OnUpdateMainRibbonItem)

	// 진단 탭
	ON_UPDATE_COMMAND_UI(ID_MAIN_RIBBON_BUTTON_analysis, &CMainFrame::OnUpdateMainRibbonAnalysis)
END_MESSAGE_MAP()

// CMainFrame 생성/소멸

CMainFrame::CMainFrame() noexcept :
	__parentSplitterWnd(.7f, 1.f)
{}

CMainFrame::~CMainFrame()
{}

int CMainFrame::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CFrameWndEx::OnCreate(lpCreateStruct) == -1)
		return -1;

	BOOL bNameValid;

	// 리본 초기화 완료
	m_wndRibbonBar.Create(this);
	m_wndRibbonBar.LoadFromResource(IDR_MAIN_RIBBON);
	m_wndRibbonBar.DeleteDropdown();

	// 리본 객체 레퍼런스를 얻어옴.
	__getRibbonControlReferences();

	if (!m_wndStatusBar.Create(this))
	{
		TRACE0("상태 표시줄을 만들지 못했습니다.\n");
		return -1;      // 만들지 못했습니다.
	}

	CString strTitlePane1;
	CString strTitlePane2;
	bNameValid = strTitlePane1.LoadString(IDS_STATUS_PANE1);
	ASSERT(bNameValid);
	bNameValid = strTitlePane2.LoadString(IDS_STATUS_PANE2);
	ASSERT(bNameValid);
	m_wndStatusBar.AddElement(new CMFCRibbonStatusBarPane(ID_STATUSBAR_PANE1, strTitlePane1, TRUE), strTitlePane1);
	m_wndStatusBar.AddExtendedElement(new CMFCRibbonStatusBarPane(ID_STATUSBAR_PANE2, strTitlePane2, TRUE), strTitlePane2);

	// Visual Studio 2005 스타일 도킹 창 동작을 활성화합니다.
	CDockingManager::SetDockingMode(DT_SMART);
	// Visual Studio 2005 스타일 도킹 창 자동 숨김 동작을 활성화합니다.
	EnableAutoHidePanes(CBRS_ALIGN_ANY);

	// 모든 사용자 인터페이스 요소를 그리는 데 사용하는 비주얼 관리자를 설정합니다.
	CMFCVisualManager::SetDefaultManager(RUNTIME_CLASS(CMFCVisualManagerOffice2007));

	// 비주얼 관리자에서 사용하는 비주얼 스타일을 설정합니다.
	CMFCVisualManagerOffice2007::SetStyle(CMFCVisualManagerOffice2007::Office2007_Silver);

	// TODO: 여기에 추가적인 초기화 내용을 작성합니다.
	__sliceViewFilterDlg.Create(IDD_DIALOG_SLICE_FILTER, this);
	__volumeRenderingFilterDlg.Create(IDD_DIALOG_VOLUME_RENDERING_FILTER, this);

	EventBroadcaster &eventBroadcaster = System::getSystemContents().getEventBroadcaster();
	eventBroadcaster.addLoginSuccessListener(*this);
	eventBroadcaster.addVolumeLoadedListener(*this);
	eventBroadcaster.addInitSlicingPointListener(*this);
	eventBroadcaster.addInitSliceViewAnchorListener(*this);
	eventBroadcaster.addUpdateSlicingPointFromViewListener(*this);
	eventBroadcaster.addUpdateAnchorFromViewListener(*this);
	eventBroadcaster.addInitLightListener(*this);

	return 0;
}

BOOL CMainFrame::PreCreateWindow(CREATESTRUCT& cs)
{
	if( !CFrameWndEx::PreCreateWindow(cs) )
		return FALSE;
	// TODO: CREATESTRUCT cs를 수정하여 여기에서
	//  Window 클래스 또는 스타일을 수정합니다.

	// 타이틀을 변경한다.
	cs.style &= ~FWS_ADDTOTITLE;
	cs.lpszName = TEXT("IntelliCAD");

	return TRUE;
}

// CMainFrame 진단

#ifdef _DEBUG
void CMainFrame::AssertValid() const
{
	CFrameWndEx::AssertValid();
}

void CMainFrame::Dump(CDumpContext& dc) const
{
	CFrameWndEx::Dump(dc);
}
#endif //_DEBUG


// CMainFrame 메시지 처리기
BOOL CMainFrame::OnCreateClient(LPCREATESTRUCT lpcs, CCreateContext* pContext)
{
	// 1 * 2 정적 분할 윈도우를 만든다.
	__parentSplitterWnd.CreateStatic(this, 1, 2);

	// 좌측 윈도우를 2 * 2로 분할한다.
	__childSplitterWnd.CreateStatic(
		&__parentSplitterWnd, 2, 2,
		(WS_CHILD | WS_VISIBLE | WS_BORDER),
		__parentSplitterWnd.IdFromRowCol(0, 0));

	// 좌측 윈도우에 렌더링 뷰를 할당한다.
	__childSplitterWnd.CreateView(0, 0, RUNTIME_CLASS(CSliceView), {}, pContext);
	__childSplitterWnd.CreateView(0, 1, RUNTIME_CLASS(CSliceView), {}, pContext);
	__childSplitterWnd.CreateView(1, 0, RUNTIME_CLASS(CSliceView), {}, pContext);
	__childSplitterWnd.CreateView(1, 1, RUNTIME_CLASS(CVolumeRenderingView), {}, pContext);

	// 좌측 윈도우 View를 구분하기 위하여 인덱스를 설정한다.

	// Top-Left
	CSliceView &topView = *__childSplitterWnd.getChildView<CSliceView>(0, 0);

	// Top-Right
	CSliceView &frontView = *__childSplitterWnd.getChildView<CSliceView>(0, 1);

	// Bottom-Left
	CSliceView &rightView = *__childSplitterWnd.getChildView<CSliceView>(1, 0);

	// Bottom-Right
	CVolumeRenderingView &volumeRenderingView = *__childSplitterWnd.getChildView<CVolumeRenderingView>(1, 1);

	// 렌더링 뷰 초기화
	topView.init(0, SliceAxis::TOP);
	frontView.init(1, SliceAxis::FRONT);
	rightView.init(2, SliceAxis::RIGHT);
	volumeRenderingView.init(3);

	__parentSplitterWnd.splitted = true;
	__childSplitterWnd.splitted = true;

	// 우측 윈도우에 Inspecter 뷰를 할당한다.
	__parentSplitterWnd.CreateView(0, 1, RUNTIME_CLASS(CInspecterView), {}, pContext);

	return true;
	// return CFrameWndEx::OnCreateClient(lpcs, pContext);
}

void CMainFrame::__openFile(const CString &path)
{
	const tstring PATH = Parser::CString$tstring(path);
	const VolumeData RESULT = VolumeReader::readMetaImage(PATH);

	if (!RESULT.pBuffer)
	{
		AfxMessageBox(_T("해당 경로에 위치한 파일을 읽을 수 없습니다."), MB_ICONERROR);
		return;
	}

	System::getSystemContents().getRenderingEngine().loadVolume(RESULT);
	
	AfxGetApp()->AddToRecentFileList(path);
}

void CMainFrame::__getRibbonControlReferences()
{
	__pRibbonSlicingPointX =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_slicingPointX));

	__pRibbonSlicingPointY =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_slicingPointY));

	__pRibbonSlicingPointZ =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_slicingPointZ));

	__pRibbonAnchorAdjXArr[SliceAxis::TOP] =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_anchorAdjX_top));

	__pRibbonAnchorAdjYArr[SliceAxis::TOP] =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_anchorAdjY_top));

	__pRibbonAnchorAdjXArr[SliceAxis::FRONT] =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_anchorAdjX_front));

	__pRibbonAnchorAdjYArr[SliceAxis::FRONT] =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_anchorAdjY_front));

	__pRibbonAnchorAdjXArr[SliceAxis::RIGHT] =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_anchorAdjX_right));

	__pRibbonAnchorAdjYArr[SliceAxis::RIGHT] =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_anchorAdjY_right));


	// Light 1

	int lightIdx = 0;

	__pRibbonTogglelLightArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonButton, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_BUTTON_togglelLight1));

	__pRibbonLightAmbientArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonColorButton, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_COLOR_BUTTON_setLight1AmbientColor));

	__pRibbonLightDiffuseArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonColorButton, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_COLOR_BUTTON_setLight1DiffuseColor));

	__pRibbonLightSpecularArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonColorButton, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_COLOR_BUTTON_setLight1SpecularColor));

	__pRibbonLightPosXArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_setLight1XPos));

	__pRibbonLightPosYArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_setLight1YPos));

	__pRibbonLightPosZArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_setLight1ZPos));

	
	// Light 2

	lightIdx++;

	__pRibbonTogglelLightArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonButton, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_BUTTON_togglelLight2));

	__pRibbonLightAmbientArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonColorButton, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_COLOR_BUTTON_setLight2AmbientColor));

	__pRibbonLightDiffuseArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonColorButton, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_COLOR_BUTTON_setLight2DiffuseColor));

	__pRibbonLightSpecularArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonColorButton, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_COLOR_BUTTON_setLight2SpecularColor));

	__pRibbonLightPosXArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_setLight2XPos));

	__pRibbonLightPosYArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_setLight2YPos));

	__pRibbonLightPosZArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_setLight2ZPos));


	// Light 3

	lightIdx++;

	__pRibbonTogglelLightArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonButton, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_BUTTON_togglelLight3));

	__pRibbonLightAmbientArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonColorButton, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_COLOR_BUTTON_setLight3AmbientColor));

	__pRibbonLightDiffuseArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonColorButton, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_COLOR_BUTTON_setLight3DiffuseColor));

	__pRibbonLightSpecularArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonColorButton, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_COLOR_BUTTON_setLight3SpecularColor));

	__pRibbonLightPosXArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_setLight3XPos));

	__pRibbonLightPosYArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_setLight3YPos));

	__pRibbonLightPosZArr[lightIdx] =
		DYNAMIC_DOWNCAST(CMFCRibbonEdit, m_wndRibbonBar.FindByID(ID_MAIN_RIBBON_EDIT_setLight3ZPos));
}

void CMainFrame::__updateSlicingPoint() const
{
	if (!__volumeLoaded)
		return;

	CString tmp;

	tmp = __pRibbonSlicingPointX->GetEditText();
	const float POINT_X = static_cast<float>(_ttof(tmp));

	tmp = __pRibbonSlicingPointY->GetEditText();
	const float POINT_Y = static_cast<float>(_ttof(tmp));

	tmp = __pRibbonSlicingPointZ->GetEditText();
	const float POINT_Z = static_cast<float>(_ttof(tmp));

	System::getSystemContents().
		getRenderingEngine().imageProcessor.setSlicingPoint({ POINT_X, POINT_Y, POINT_Z });

	EventBroadcaster &eventBroadcaster = System::getSystemContents().getEventBroadcaster();
	eventBroadcaster.notifyRequestScreenUpdate(RenderingScreenType::SLICE_TOP);
	eventBroadcaster.notifyRequestScreenUpdate(RenderingScreenType::SLICE_FRONT);
	eventBroadcaster.notifyRequestScreenUpdate(RenderingScreenType::SLICE_RIGHT);
}

static RenderingScreenType __mapAxisToScreenType(const SliceAxis axis)
{
	RenderingScreenType retVal = RenderingScreenType::SLICE_TOP;

	switch (axis)
	{
	case SliceAxis::TOP:
		retVal = RenderingScreenType::SLICE_TOP;
		break;

	case SliceAxis::FRONT:
		retVal = RenderingScreenType::SLICE_FRONT;
		break;

	case SliceAxis::RIGHT:
		retVal = RenderingScreenType::SLICE_RIGHT;
		break;
	}

	return retVal;
}

void CMainFrame::__updateAnchorAdj(const SliceAxis axis) const
{
	if (!__volumeLoaded)
		return;

	CString tmp;

	tmp = __pRibbonAnchorAdjXArr[axis]->GetEditText();
	const float ADJ_X = static_cast<float>(_ttof(tmp));

	tmp = __pRibbonAnchorAdjYArr[axis]->GetEditText();
	const float ADJ_Y = static_cast<float>(_ttof(tmp));

	System::getSystemContents().
		getRenderingEngine().imageProcessor.setAnchorAdj(ADJ_X, ADJ_Y, axis);

	const RenderingScreenType TARGET_SCREEN = __mapAxisToScreenType(axis);

	System::getSystemContents().
		getEventBroadcaster().notifyRequestScreenUpdate(TARGET_SCREEN);
}

void CMainFrame::__initRibbonEditAnchorAdj(const SliceAxis axis) const
{
	__pRibbonAnchorAdjXArr[axis]->SetEditText(_T("0.000000"));
	__pRibbonAnchorAdjYArr[axis]->SetEditText(_T("0.000000"));
}

void CMainFrame::__onMainRibbonInitAnchorAdj(const SliceAxis axis) const
{
	__initRibbonEditAnchorAdj(axis);

	System::getSystemContents().
		getRenderingEngine().imageProcessor.setAnchorAdj(0.f, 0.f, axis);

	System::getSystemContents().
		getEventBroadcaster().notifyRequestScreenUpdate(__mapAxisToScreenType(axis));
}

void CMainFrame::__onMainRibbonInitVolumeFilter(const ColorChannelType colorType) const
{
	System::getSystemContents().
		getRenderingEngine().volumeRenderer.initTransferFunction(colorType);
}

void CMainFrame::__toggleLight(const int index)
{
	RenderingEngine::VolumeRenderer& volumeRenderer = System::getSystemContents().getRenderingEngine().volumeRenderer;

	volumeRenderer.toggleLighting(index);

	if (volumeRenderer.getLight(index).enabled)
		__pRibbonTogglelLightArr[index]->SetImageIndex(2, true);
	else
		__pRibbonTogglelLightArr[index]->SetImageIndex(1, true);

	System::getSystemContents().getEventBroadcaster().notifyRequestScreenUpdate(RenderingScreenType::VOLUME_RENDERING);
}

void CMainFrame::__setLightAmbientColor(const int index)
{
	RenderingEngine::VolumeRenderer& volumeRenderer = System::getSystemContents().getRenderingEngine().volumeRenderer;
	Color<float> COLOR = Parser::COLORREF$Color(__pRibbonLightAmbientArr[index]->GetColor());

	volumeRenderer.setLightAmbient(index, COLOR);
	System::getSystemContents().getEventBroadcaster().
		notifyRequestScreenUpdate(RenderingScreenType::VOLUME_RENDERING);
}

void CMainFrame::__setLightDiffuseColor(const int index)
{
	RenderingEngine::VolumeRenderer& volumeRenderer = System::getSystemContents().getRenderingEngine().volumeRenderer;
	Color<float> COLOR = Parser::COLORREF$Color(__pRibbonLightDiffuseArr[index]->GetColor());

	volumeRenderer.setLightDiffuse(index, COLOR);
	System::getSystemContents().getEventBroadcaster().
		notifyRequestScreenUpdate(RenderingScreenType::VOLUME_RENDERING);
}

void CMainFrame::__setLightSpecularColor(const int index)
{
	RenderingEngine::VolumeRenderer& volumeRenderer = System::getSystemContents().getRenderingEngine().volumeRenderer;
	Color<float> COLOR = Parser::COLORREF$Color(__pRibbonLightSpecularArr[index]->GetColor());

	volumeRenderer.setLightSpecular(index, COLOR);
	System::getSystemContents().getEventBroadcaster().
		notifyRequestScreenUpdate(RenderingScreenType::VOLUME_RENDERING);
}

void CMainFrame::__setLightXPos(const int index)
{
	RenderingEngine::VolumeRenderer& volumeRenderer = System::getSystemContents().getRenderingEngine().volumeRenderer;
	const float X = Parser::CString$float<float>(__pRibbonLightPosXArr[index]->GetEditText());

	volumeRenderer.setLightXPos(index, X);
	System::getSystemContents().getEventBroadcaster().
		notifyRequestScreenUpdate(RenderingScreenType::VOLUME_RENDERING);
}

void CMainFrame::__setLightYPos(const int index)
{
	RenderingEngine::VolumeRenderer& volumeRenderer = System::getSystemContents().getRenderingEngine().volumeRenderer;
	const float Y = Parser::CString$float<float>(__pRibbonLightPosYArr[index]->GetEditText());

	volumeRenderer.setLightYPos(index, Y);
	System::getSystemContents().getEventBroadcaster().
		notifyRequestScreenUpdate(RenderingScreenType::VOLUME_RENDERING);
}

void CMainFrame::__setLightZPos(const int index)
{
	RenderingEngine::VolumeRenderer& volumeRenderer = System::getSystemContents().getRenderingEngine().volumeRenderer;
	const float Z = Parser::CString$float<float>(__pRibbonLightPosZArr[index]->GetEditText());

	volumeRenderer.setLightXPos(index, Z);
	System::getSystemContents().getEventBroadcaster().
		notifyRequestScreenUpdate(RenderingScreenType::VOLUME_RENDERING);
}

void CMainFrame::OnFileOpen()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	const TCHAR *const szFilter = _T("MetaImage (*.mhd) |*.mhd|");
	CFileDialog fileDlg(true, _T("MetaImage (*.mhd)"), nullptr, OFN_FILEMUSTEXIST, szFilter);

	if (fileDlg.DoModal() == IDOK)
		__openFile(fileDlg.GetPathName());
}


void CMainFrame::OnButtonCloudService()
{
	RemoteAccessAuthorizer &accessAuthorizer =
		System::getSystemContents().getRemoteAccessAuthorizer();

	if (!accessAuthorizer.isAuthorized())
		if (__loginDlg.DoModal() == LOGIN_DLG_LOGIN_SUCCESS)
			MessageBox(_T("로그인에 성공하였습니다."), _T("로그인 성공"), MB_ICONINFORMATION);
}

void CMainFrame::OnMainRibbonButtonSetSliceFilter()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	if (__sliceViewFilterDlg.IsWindowVisible())
		__sliceViewFilterDlg.ShowWindow(SW_RESTORE);
	else
		__sliceViewFilterDlg.ShowWindow(SW_SHOWDEFAULT);
}

void CMainFrame::OnMainRibbonButtonInitSliceFilter()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	System::getSystemContents().
		getRenderingEngine().imageProcessor.initTransferFunction();
}

void CMainFrame::OnOpenMRUFile(UINT nID)
{
	int index = -1;
	switch (nID)
	{
	case ID_FILE_MRU_FILE1:
		index = 0;
		break;

	case ID_FILE_MRU_FILE2:
		index = 1;
		break;

	case ID_FILE_MRU_FILE3:
		index = 2;
		break;

	case ID_FILE_MRU_FILE4:
		index = 3;
		break;
	}

	const CString &PATH =
		static_cast<CIntelliCADApp *>(AfxGetApp())->getRecentFileName(index);

	__openFile(PATH);
}

void CMainFrame::onVolumeLoaded(const VolumeMeta &volumeMeta)
{
	__volumeLoaded = true;

	System::getInstance().printLog(_T("이미지 ") + volumeMeta.fileName + _T("를 불러왔습니다."));
}

void CMainFrame::onInitSlicingPoint(const Point3D &slicingPoint)
{
	CString tmp;

	// Slicing point - x
	tmp.Format(_T("%f"), slicingPoint.x);
	__pRibbonSlicingPointX->SetEditText(tmp);

	// Slicing point - y
	tmp.Format(_T("%f"), slicingPoint.y);
	__pRibbonSlicingPointY->SetEditText(tmp);

	// Slicing point - z
	tmp.Format(_T("%f"), slicingPoint.z);
	__pRibbonSlicingPointZ->SetEditText(tmp);
}

void CMainFrame::onInitSliceViewAnchor(const SliceAxis axis)
{
	__initRibbonEditAnchorAdj(axis);
}

void CMainFrame::onLoginSuccess(const Account &account)
{
	CString msg;
	msg.Format(_T("아이디 %s로 로그인 되었습니다."), account.id);

	System::getInstance().printLog(Parser::CString$tstring(msg));
}

void CMainFrame::onUpdateSlicingPointFromView()
{
	const Point3D &SLICING_POINT =
		System::getSystemContents().getRenderingEngine().imageProcessor.getSlicingPoint();

	CString tmp;

	// Slicing point - x
	tmp.Format(_T("%f"), SLICING_POINT.x);
	__pRibbonSlicingPointX->SetEditText(tmp);

	// Slicing point - y
	tmp.Format(_T("%f"), SLICING_POINT.y);
	__pRibbonSlicingPointY->SetEditText(tmp);

	// Slicing point - z
	tmp.Format(_T("%f"), SLICING_POINT.z);
	__pRibbonSlicingPointZ->SetEditText(tmp);
}

void CMainFrame::onUpdateAnchorFromView(const SliceAxis axis)
{
	const Point2D &ANCHOR_ADJ =
		System::getSystemContents().getRenderingEngine().imageProcessor.getAnchorAdj(axis);

	CString tmp;

	tmp.Format(_T("%f"), ANCHOR_ADJ.x);
	__pRibbonAnchorAdjXArr[axis]->SetEditText(tmp);

	tmp.Format(_T("%f"), ANCHOR_ADJ.y);
	__pRibbonAnchorAdjYArr[axis]->SetEditText(tmp);
}

void CMainFrame::onInitLight()
{
	RenderingEngine::VolumeRenderer& volumeRenderer = System::getSystemContents().getRenderingEngine().volumeRenderer;

	CString tmp;

	for (int i = 0; i < 3; i++)
	{
		const Light &light = volumeRenderer.getLight(i);

		__pRibbonTogglelLightArr[i]->SetImageIndex(1, true);
		__pRibbonLightAmbientArr[i]->SetColor(Parser::Color$COLORREF(light.ambient));
		__pRibbonLightDiffuseArr[i]->SetColor(Parser::Color$COLORREF(light.diffuse));
		__pRibbonLightSpecularArr[i]->SetColor(Parser::Color$COLORREF(light.specular));

		tmp.Format(_T("%f"), light.position.x);
		__pRibbonLightPosXArr[i]->SetEditText(tmp);

		tmp.Format(_T("%f"), light.position.y);
		__pRibbonLightPosYArr[i]->SetEditText(tmp);

		tmp.Format(_T("%f"), light.position.z);
		__pRibbonLightPosZArr[i]->SetEditText(tmp);
	}
}

void CMainFrame::OnMainRibbonUpdateSlicingPoint()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	__updateSlicingPoint();
}

void CMainFrame::OnMainRibbonInitSlicinigPoint()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	if (!__volumeLoaded)
		return;

	System::getSystemContents().
		getRenderingEngine().imageProcessor.setSlicingPointAdj({ 0.f, 0.f, 0.f });

	EventBroadcaster &eventBroadcaster = System::getSystemContents().getEventBroadcaster();
	eventBroadcaster.notifyRequestScreenUpdate(RenderingScreenType::SLICE_TOP);
	eventBroadcaster.notifyRequestScreenUpdate(RenderingScreenType::SLICE_FRONT);
	eventBroadcaster.notifyRequestScreenUpdate(RenderingScreenType::SLICE_RIGHT);
}

void CMainFrame::OnMainRibbonUpdateAnchorAdj_top()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	__updateAnchorAdj(SliceAxis::TOP);
}

void CMainFrame::OnMainRibbonInitAnchorAdj_top()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	if (!__volumeLoaded)
		return;

	__onMainRibbonInitAnchorAdj(SliceAxis::TOP);
}

void CMainFrame::OnMainRibbonUpdateAnchorAdj_front()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	__updateAnchorAdj(SliceAxis::FRONT);
}

void CMainFrame::OnMainRibbonInitAnchorAdj_front()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	if (!__volumeLoaded)
		return;

	__onMainRibbonInitAnchorAdj(SliceAxis::FRONT);
}

void CMainFrame::OnMainRibbonUpdateAnchorAdj_right()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	__updateAnchorAdj(SliceAxis::RIGHT);
}

void CMainFrame::OnMainRibbonInitAnchorAdj_right()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	if (!__volumeLoaded)
		return;

	__onMainRibbonInitAnchorAdj(SliceAxis::RIGHT);
}

void CMainFrame::OnMainRibbonSetVolumeFilter()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	if (__volumeRenderingFilterDlg.IsWindowVisible())
		__volumeRenderingFilterDlg.ShowWindow(SW_RESTORE);
	else
		__volumeRenderingFilterDlg.ShowWindow(SW_SHOWDEFAULT);
}

void CMainFrame::OnMainRibbonInitVolumeFilterRed()
{
	__onMainRibbonInitVolumeFilter(ColorChannelType::RED);
}

void CMainFrame::OnMainRibbonInitVolumeFilterGreen()
{
	__onMainRibbonInitVolumeFilter(ColorChannelType::GREEN);
}

void CMainFrame::OnMainRibbonInitVolumeFilterBlue()
{
	__onMainRibbonInitVolumeFilter(ColorChannelType::BLUE);
}

void CMainFrame::OnMainRibbonInitVolumeFilterAlpha()
{
	__onMainRibbonInitVolumeFilter(ColorChannelType::ALPHA);
}

void CMainFrame::OnMainRibbonInitVolumeFilterAll()
{
	__onMainRibbonInitVolumeFilter(ColorChannelType::ALL);
}

void CMainFrame::OnMainRibbonButtonToggleLight1()
{
	__toggleLight(0);
}

void CMainFrame::OnMainRibbonButtonSetLight1AmbientColor()
{
	__setLightAmbientColor(0);
}

void CMainFrame::OnMainRibbonButtonSetLight1DiffuseColor()
{
	__setLightDiffuseColor(0);
}

void CMainFrame::OnMainRibbonButtonSetLight1SpecularColor()
{
	__setLightSpecularColor(0);
}

void CMainFrame::OnMainRibbonButtonSetLight1XPos()
{
	__setLightXPos(0);
}

void CMainFrame::OnMainRibbonButtonSetLight1YPos()
{
	__setLightYPos(0);
}

void CMainFrame::OnMainRibbonButtonSetLight1ZPos()
{
	__setLightZPos(0);
}

void CMainFrame::OnMainRibbonButtonToggleLight2()
{
	__toggleLight(1);
}

void CMainFrame::OnMainRibbonButtonSetLight2AmbientColor()
{
	__setLightAmbientColor(1);
}

void CMainFrame::OnMainRibbonButtonSetLight2DiffuseColor()
{
	__setLightDiffuseColor(1);
}

void CMainFrame::OnMainRibbonButtonSetLight2SpecularColor()
{
	__setLightSpecularColor(1);
}

void CMainFrame::OnMainRibbonButtonSetLight2XPos()
{
	__setLightXPos(1);
}

void CMainFrame::OnMainRibbonButtonSetLight2YPos()
{
	__setLightYPos(1);
}

void CMainFrame::OnMainRibbonButtonSetLight2ZPos()
{
	__setLightZPos(1);
}

void CMainFrame::OnMainRibbonButtonToggleLight3()
{
	__toggleLight(2);
}

void CMainFrame::OnMainRibbonButtonSetLight3AmbientColor()
{
	__setLightAmbientColor(2);
}

void CMainFrame::OnMainRibbonButtonSetLight3DiffuseColor()
{
	__setLightDiffuseColor(2);
}

void CMainFrame::OnMainRibbonButtonSetLight3SpecularColor()
{
	__setLightSpecularColor(2);
}

void CMainFrame::OnMainRibbonButtonSetLight3XPos()
{
	__setLightXPos(2);
}

void CMainFrame::OnMainRibbonButtonSetLight3YPos()
{
	__setLightYPos(2);
}

void CMainFrame::OnMainRibbonButtonSetLight3ZPos()
{
	__setLightZPos(2);
}

void CMainFrame::OnUpdateMainRibbonItem(CCmdUI *pCmdUI)
{
	// TODO: 여기에 명령 업데이트 UI 처리기 코드를 추가합니다.
	pCmdUI->Enable(__volumeLoaded);
}

void CMainFrame::OnUpdateMainRibbonAnalysis(CCmdUI *pCmdUI)
{
	pCmdUI->Enable(__volumeLoaded && !__analyzing);
}

void CMainFrame::OnMainRibbonAnalysis()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.

}
