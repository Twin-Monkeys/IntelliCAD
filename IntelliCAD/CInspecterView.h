#pragma once

#include "stdafx.h"
#include "CImageInfoDialog.h"
#include "CServerInteractionActiveDlg.h"
#include "CServerInteractionInactiveDlg.h"
#include "CLogDialog.h"
#include "LoginSuccessListener.h"

class CInspecterView : public CFormView, public LoginSuccessListener
{
	DECLARE_DYNCREATE(CInspecterView)
	DECLARE_MESSAGE_MAP()

public:
	CInspecterView();
	virtual ~CInspecterView();

#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG_INSPECTOR };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);

private:
	bool __initialized = false;

	CTabCtrl __ddx_tab;
	CImageInfoDialog __imageInfoDlg;
	CServerInteractionActiveDlg __serverActiveDlg;
	CServerInteractionInactiveDlg __serverInactiveDlg;
	CLogDialog __logDlg;

	CWnd *__pPrevTabDlg = nullptr;
	bool __serverActivated = false;

	void __recalcLayout();
	void __updateTabDlg();

public:
	virtual void OnInitialUpdate();
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnTcnSelchangeInspectorTab(NMHDR *pNMHDR, LRESULT *pResult);

	virtual void onLoginSuccess(const Account &account) override;
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
};
