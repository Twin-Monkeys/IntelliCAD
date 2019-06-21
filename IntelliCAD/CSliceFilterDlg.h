#pragma once

#include "MTFChartBuilder.h"
#include "InitSliceTransferFunctionListener.h"

// CSliceFilterDlg 대화 상자

class CSliceFilterDlg :
	public CDialogEx, public InitSliceTransferFunctionListener
{
	DECLARE_DYNAMIC(CSliceFilterDlg)
	DECLARE_MESSAGE_MAP()

public:
	CSliceFilterDlg(CWnd* pParent = nullptr);   // 표준 생성자입니다.
	virtual ~CSliceFilterDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG_SLICE_FILTER };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

private:
	CChartViewer __ddx_chartViewer;
	MTFChartBuilder __chartBuilder;

	bool __activated = false;
	bool __chartViewLButtonDown = false;

	HACCEL __hAccel = nullptr;

	void __init();
	void __render();

public:
	bool isActive() const;

	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnPaint();

	void OnLButtonDownChartView();
	void OnLButtonUpChartView();
	void OnMouseMovePlotArea();

	virtual BOOL PreTranslateMessage(MSG* pMsg);
	afx_msg void OnGetMinMaxInfo(MINMAXINFO* lpMMI);

	virtual void onInitSliceTransferFunction() override;
	afx_msg void OnMenuSliceFilterEditInitSliceFilter();
	afx_msg BOOL OnEraseBkgnd(CDC* pDC);
	virtual BOOL OnInitDialog();

	afx_msg void OnDestroy();
};
