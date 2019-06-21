#pragma once

#include "CTFChartBuilder.h"
#include "InitVolumeTransferFunctionListener.h"

// CVolumeRenderingFilterDlg 대화 상자

class CVolumeRenderingFilterDlg : public CDialogEx, public InitVolumeTransferFunctionListener
{
	DECLARE_DYNAMIC(CVolumeRenderingFilterDlg)
	DECLARE_MESSAGE_MAP()

public:
	CVolumeRenderingFilterDlg(CWnd* pParent = nullptr);   // 표준 생성자입니다.
	virtual ~CVolumeRenderingFilterDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG_VOLUME_RENDERING_FILTER };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

private:
	CChartViewer __ddx_chartViewer;
	CTFChartBuilder __chartBuilder;

	bool __activated = false;
	bool __chartViewLButtonDown = false;

	HACCEL __hAccel = nullptr;

	void __init(const ColorChannelType colorType);
	void __render();

	void __setMenuItemChecked(const UINT nIDCheckItem, const bool checked);

public:
	bool isActive() const;

	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnPaint();

	void OnLButtonDownChartView();
	void OnLButtonUpChartView();
	void OnMouseMovePlotArea();

	virtual BOOL PreTranslateMessage(MSG* pMsg);
	afx_msg void OnGetMinMaxInfo(MINMAXINFO* lpMMI);

	virtual void onInitVolumeTransferFunction(const ColorChannelType colorType) override;

	afx_msg void OnMenuInitRedFilter();
	afx_msg void OnMenuInitGreenFilter();
	afx_msg void OnMenuInitBlueFilter();
	afx_msg void OnMenuInitAllFilter();
	afx_msg void OnMenuInitAlphaFilter();
	afx_msg void OnMenuToggleTargetFilterRed();
	afx_msg void OnMenuToggleTargetFilterGreen();
	afx_msg void OnMenuToggleTargetFilterBlue();
	afx_msg void OnMenuToggleTargetFilterAlpha();
	afx_msg void OnMenuSelectAllTargetFilter();
	afx_msg void OnMenuSelectNoneTargetFilter();
	afx_msg BOOL OnEraseBkgnd(CDC* pDC);
	virtual BOOL OnInitDialog();

	afx_msg void OnDestroy();
};
