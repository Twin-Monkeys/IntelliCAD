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

// MainFrm.h: CMainFrame 클래스의 인터페이스
//

#pragma once

#include "CCustomSplitterWnd.h"
#include "CCustomRibbonBar.h"
#include "CSliceFilterDlg.h"
#include "CVolumeRenderingFilterDlg.h"
#include "CLoginDlg.h"
#include "LoginSuccessListener.h"
#include "VolumeLoadedListener.h"
#include "InitSlicingPointListener.h"
#include "InitSliceViewAnchorListener.h"
#include "UpdateSlicingPointFromViewListener.h"
#include "UpdateAnchorFromViewListener.h"
#include "InitLightListener.h"

class CMainFrame :
	public CFrameWndEx, public LoginSuccessListener, public VolumeLoadedListener,
	public InitSlicingPointListener, public InitSliceViewAnchorListener,
	public UpdateSlicingPointFromViewListener, public UpdateAnchorFromViewListener,
	public InitLightListener
{
	
protected: // serialization에서만 만들어집니다.
	CMainFrame() noexcept;
	DECLARE_DYNCREATE(CMainFrame)

// 특성입니다.
public:

// 작업입니다.
public:

// 재정의입니다.
public:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);

// 구현입니다.
public:
	virtual ~CMainFrame();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:  // 컨트롤 모음이 포함된 멤버입니다.
	CCustomRibbonBar     m_wndRibbonBar;
	CMFCRibbonApplicationButton m_MainButton;
	CMFCToolBarImages m_PanelImages;
	CMFCRibbonStatusBar  m_wndStatusBar;

// 생성된 메시지 맵 함수
protected:
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	DECLARE_MESSAGE_MAP()

private:
	CCustomSplitterWnd __parentSplitterWnd;
	CCustomSplitterWnd __childSplitterWnd;

	CSliceFilterDlg __sliceViewFilterDlg;
	CVolumeRenderingFilterDlg __volumeRenderingFilterDlg;
	CLoginDlg __loginDlg;

	bool __volumeLoaded = false;
	bool __analyzing = false;

	CMFCRibbonEdit *__pRibbonSlicingPointX = nullptr;
	CMFCRibbonEdit *__pRibbonSlicingPointY = nullptr;
	CMFCRibbonEdit *__pRibbonSlicingPointZ = nullptr;

	CMFCRibbonEdit *__pRibbonAnchorAdjXArr[3] = { nullptr, nullptr, nullptr };
	CMFCRibbonEdit *__pRibbonAnchorAdjYArr[3] = { nullptr, nullptr, nullptr };
	
	CMFCRibbonButton *__pRibbonTogglelLightArr[3] = { nullptr, nullptr, nullptr };
	CMFCRibbonColorButton *__pRibbonLightAmbientArr[3] = { nullptr, nullptr, nullptr };
	CMFCRibbonColorButton *__pRibbonLightDiffuseArr[3] = { nullptr, nullptr, nullptr };
	CMFCRibbonColorButton *__pRibbonLightSpecularArr[3] = { nullptr, nullptr, nullptr };
	CMFCRibbonEdit *__pRibbonLightPosXArr[3] = { nullptr, nullptr, nullptr };
	CMFCRibbonEdit *__pRibbonLightPosYArr[3] = { nullptr, nullptr, nullptr };
	CMFCRibbonEdit *__pRibbonLightPosZArr[3] = { nullptr, nullptr, nullptr };

	void __openFile(const CString &path);

	void __getRibbonControlReferences();
	void __updateSlicingPoint() const;

	void __updateAnchorAdj(const SliceAxis axis) const;
	void __initRibbonEditAnchorAdj(const SliceAxis axis) const;
	void __onMainRibbonInitAnchorAdj(const SliceAxis axis) const;
	void __onMainRibbonInitVolumeFilter(const ColorChannelType colorType) const;

	void __toggleLight(const int lightIdx);
	void __setLightAmbientColor(const int index);
	void __setLightDiffuseColor(const int index);
	void __setLightSpecularColor(const int index);
	void __setLightXPos(const int index);
	void __setLightYPos(const int index);
	void __setLightZPos(const int index);

public:
	virtual BOOL OnCreateClient(LPCREATESTRUCT lpcs, CCreateContext* pContext);
	afx_msg void OnFileOpen();
	afx_msg void OnButtonCloudService();
	afx_msg void OnOpenMRUFile(UINT nID);

	afx_msg void OnMainRibbonButtonSetSliceFilter();
	afx_msg void OnMainRibbonButtonInitSliceFilter();

	virtual void onVolumeLoaded(const VolumeMeta &volumeMeta) override;
	virtual void onInitSlicingPoint(const Point3D &slicingPoint) override;
	virtual void onInitSliceViewAnchor(const SliceAxis axis) override;
	virtual void onLoginSuccess(const Account &account) override;
	virtual void onUpdateSlicingPointFromView() override;
	virtual void onUpdateAnchorFromView(const SliceAxis axis) override;
	virtual void onInitLight() override;

	afx_msg void OnMainRibbonUpdateSlicingPoint();
	afx_msg void OnMainRibbonInitSlicinigPoint();

	afx_msg void OnMainRibbonUpdateAnchorAdj_top();
	afx_msg void OnMainRibbonInitAnchorAdj_top();

	afx_msg void OnMainRibbonUpdateAnchorAdj_front();
	afx_msg void OnMainRibbonInitAnchorAdj_front();

	afx_msg void OnMainRibbonUpdateAnchorAdj_right();
	afx_msg void OnMainRibbonInitAnchorAdj_right();

	afx_msg void OnMainRibbonSetVolumeFilter();

	afx_msg void OnMainRibbonInitVolumeFilterRed();
	afx_msg void OnMainRibbonInitVolumeFilterGreen();
	afx_msg void OnMainRibbonInitVolumeFilterBlue();
	afx_msg void OnMainRibbonInitVolumeFilterAlpha();
	afx_msg void OnMainRibbonInitVolumeFilterAll();

	afx_msg void OnMainRibbonButtonToggleLight1();
	afx_msg void OnMainRibbonButtonSetLight1AmbientColor();
	afx_msg void OnMainRibbonButtonSetLight1DiffuseColor();
	afx_msg void OnMainRibbonButtonSetLight1SpecularColor();
	afx_msg void OnMainRibbonButtonSetLight1XPos();
	afx_msg void OnMainRibbonButtonSetLight1YPos();
	afx_msg void OnMainRibbonButtonSetLight1ZPos();

	afx_msg void OnMainRibbonButtonToggleLight2();
	afx_msg void OnMainRibbonButtonSetLight2AmbientColor();
	afx_msg void OnMainRibbonButtonSetLight2DiffuseColor();
	afx_msg void OnMainRibbonButtonSetLight2SpecularColor();
	afx_msg void OnMainRibbonButtonSetLight2XPos();
	afx_msg void OnMainRibbonButtonSetLight2YPos();
	afx_msg void OnMainRibbonButtonSetLight2ZPos();

	afx_msg void OnMainRibbonButtonToggleLight3();
	afx_msg void OnMainRibbonButtonSetLight3AmbientColor();
	afx_msg void OnMainRibbonButtonSetLight3DiffuseColor();
	afx_msg void OnMainRibbonButtonSetLight3SpecularColor();
	afx_msg void OnMainRibbonButtonSetLight3XPos();
	afx_msg void OnMainRibbonButtonSetLight3YPos();
	afx_msg void OnMainRibbonButtonSetLight3ZPos();

	afx_msg void OnUpdateMainRibbonItem(CCmdUI *pCmdUI);
	afx_msg void OnUpdateMainRibbonAnalysis(CCmdUI *pCmdUI);
	afx_msg void OnMainRibbonAnalysis();
};